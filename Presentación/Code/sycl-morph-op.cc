handler.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
  auto global_id = item.get_global_id();
  auto group_id = item.get_group().get_group_id();
  auto local_id = item.get_local_id();
  // ...
  // Memory tiling operation omitted
  // ...
  auto tile_index_origin = local_id + sel_offset;
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
  output_accessor[global_id] = minimum;
});