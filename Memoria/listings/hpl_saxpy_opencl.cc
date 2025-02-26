using namespace HPL;

// String with the OpenCL C kernel for SAXPY
const char* saxpy_kernel = 
  "__kernel void saxpy(__global float *y, __global float *x, float alpha) {\n \
     int i = get_global_id(0);                                             \n \
     y[i] = alpha * x[i] + y[i];                                           \n \
  }";

// Function whose arguments define the kernel for HPL
void saxpy_handle(InOut< Array<float,1> > y, In< Array<float,1> > x, Float alpha) { }

int main(int argc, char **argv) {
  Array<float, 1> x(1000), y(1000);
  float alpha;
  // The vectors x and y are filled in with data (not shown)
  // Associate the kernel string with the HPL function handle
  nativeHandle(saxpy_handle, "saxpy", saxpy_kernel);
  eval(saxpy_handle)(y, x, alpha);
}
