#include <sycl/sycl.hpp>
#include <iostream>

int main() {
  auto AsyncHandler = [](sycl::exception_list exceptions) {
    for (auto& exception: exceptions) {
      std::rethrow_exception(exception);
    }
  };
  try {
    sycl::queue queue{AsyncHandler};
    // Potentially exception-prone code
    // ...
  } catch (sycl::exception& e) {
    std::cerr << "SYCL exception has occurred.\n"
              << e.what() << std::endl;
  }
}
