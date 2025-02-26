for (long row{0}; row < image.Rows(); ++row) {
  long image_row = origin + row * image.PaddedColumns();
  for (long column{0}; column < image.Columns(); ++column) {
    long pixel_index = image_row + column;
    long local_origin = pixel_index -
                        sel.CenterRow() * image.PaddedColumns() -
                        sel.CenterColumn();
    T minimum = std::numeric_limits<T>::max();
    for (long local_row{0}; local_row < sel.Rows(); ++local_row) {
      long local_image_row = local_origin + local_row * image.PaddedColumns();
      long sel_row = local_row * sel.Columns();
      for (long local_column{0}; local_column < sel.Columns(); ++local_column) {
        long local_pixel = local_image_row + local_column;
        if (sel_data[sel_row + local_column] == 1 && 
            image_data_copy[local_pixel] <= minimum) {
          minimum = image_data_copy[local_pixel];
        }
      }
    }
    image_data[pixel_index] = minimum;
  }
}
delete[] image_data_copy;