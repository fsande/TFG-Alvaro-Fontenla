handler.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
  auto global_id = item.get_global_id();
  auto group_id = item.get_group().get_group_id();
  auto local_id = item.get_local_id();
  auto global_group_offset = group_id * local_range;
  for (auto row = local_id[0]; row < tile_range[0];
       row += local_range[0]) {
    for (auto column = local_id[1]; column < tile_range[1]; 
         column += local_range[1]) {
      tile[row][column] =
        image_accessor[global_group_offset + sycl::range(row, column)];
    }
  }
  sycl::group_barrier(item.get_group());
  // ...
  // Pixel value calculation omitted
  // ...
});