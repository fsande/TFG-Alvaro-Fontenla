using namespace HPL;

// SAXPY kernel in which thread idx computes y[idx]
void saxpy(Array<float,1> y, Array<float,1> x, Float alpha) {
  y[idx] = alpha * x[idx] + y[idx];
}

int main(int argc, char **argv) {
  Array<float, 1> x(1000), y(1000);
  float alpha;
  // The vectors x and y are filled in with data (not shown)
  // Run SAXPY on an accelerator, or the CPU if no OpenCL capable accelerator is found
  eval(saxpy)(y, x, alpha);
}
