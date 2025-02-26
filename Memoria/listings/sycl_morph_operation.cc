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