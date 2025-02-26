#include <sycl/sycl.hpp>
#include <iostream>

int main() {
  sycl::queue queue{sycl::cpu_selector_v};
  std::cout << "Device: "
            << queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
}
