TemplatedFitsImage<T>& image =
  *dynamic_cast<TemplatedFitsImage<T>*>(fits_image);
TemplatedStructuringElement<T>& sel =
  *dynamic_cast<TemplatedStructuringElement<T>*>(operation_sel);
T* image_data = image.GetData();
T* image_data_copy = new T[image.PaddedTotalElements()];
std::copy(image_data, image_data + image.PaddedTotalElements(), image_data_copy);
T* sel_data = sel.GetData();
long origin = image.Padding() * image.PaddedColumns() + image.Padding();