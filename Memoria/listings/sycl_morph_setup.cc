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