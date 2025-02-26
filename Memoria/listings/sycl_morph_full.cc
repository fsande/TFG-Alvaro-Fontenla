void Operate(FitsImage* fits_image, StructuringElement* operation_sel) override {
  sycl::queue queue{sycl::gpu_selector_v};
  TemplatedFitsImage<T>& image =
    *dynamic_cast<TemplatedFitsImage<T>*>(fits_image);
  TemplatedStructuringElement<T>& sel =
    *dynamic_cast<TemplatedStructuringElement<T>*>(operation_sel);
  T* image_data = image.GetData();
  T* sel_data = sel.GetData();

  const long kTwicePadding = 2 * image.Padding();
  auto twice_padding_range = sycl::range(kTwicePadding, kTwicePadding);
  auto padding_range = sycl::range(image.Padding(), image.Padding());
  auto sel_offset =
    padding_range - sycl::range(sel.CenterRow(), sel.CenterColumn());

  { // Buffer scope
  // CG Ranges
  auto local_range = sycl::range(sel.Rows(), sel.Columns());
  int column_work_groups_amount =
    FitsUtils::DivisionCeiling(image.Columns(), local_range[1]);
  int row_work_groups_amount =
    FitsUtils::DivisionCeiling(image.Rows(), local_range[0]);
  auto global_range = sycl::range(local_range[0] * row_work_groups_amount,
                                  local_range[1] * column_work_groups_amount);
  auto nd_range = sycl::nd_range(global_range, local_range);
  // Buffer Ranges
  auto image_buffer_range =
    sycl::range(image.PaddedRows(), image.PaddedColumns());
  auto output_buffer_range =
    sycl::range(image.PaddedRows(), image.PaddedColumns());
  auto sel_buffer_range = sycl::range(sel.Rows(), sel.Columns());
  auto tile_range = local_range + twice_padding_range;
  // Buffers
  auto image_buffer = sycl::buffer{image_data, image_buffer_range};
  image_buffer.set_final_data(nullptr);
  auto output_buffer = sycl::buffer<T, 2>{output_buffer_range};
  output_buffer.set_final_data(image_data);
  auto sel_buffer = sycl::buffer{sel_data, sel_buffer_range};
  // Command Group Submission
  queue.submit([&](sycl::handler& handler) {
    sycl::accessor image_accessor{image_buffer, handler, sycl::read_only};
    sycl::accessor output_accessor{output_buffer, handler, sycl::write_only};
    sycl::accessor sel_accessor{sel_buffer, handler, sycl::read_only};
    auto tile = sycl::local_accessor<T, 2>(tile_range, handler);

    handler.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
      auto global_id = item.get_global_id();
      auto group_id = item.get_group().get_group_id();
      auto local_id = item.get_local_id();
      auto global_group_offset = group_id * local_range;
      
      // Load tile
      for (auto row = local_id[0]; row < tile_range[0]; row += local_range[0]) {
        for (auto column = local_id[1]; column < tile_range[1]; column += local_range[1]) {
          tile[row][column] = image_accessor[global_group_offset + sycl::range(row, column)];
        }
      }
      sycl::group_barrier(item.get_group());
      auto tile_index_origin = local_id + sel_offset;

      // Erode
      T minimum = std::numeric_limits<T>::max();
      for (int row = 0; row < sel_buffer_range[0]; ++row) {
        for (int column = 0; column < sel_buffer_range[1]; ++column) {
          auto tile_index = tile_index_origin + sycl::range(row, column);
          if (sel_accessor[row][column] == static_cast<T>(1) &&
              tile[tile_index] < minimum) {
            minimum = tile[tile_index];
          }
        }
      }
      // Write output
      output_accessor[global_id] = minimum;
    });
  });
  queue.wait_and_throw();
  }
}