#include <sycl/sycl.hpp>

int main() {
  int summand_a{1}, summand_b{2}, result{0};
  sycl::queue queue;
  { // Buffer scope
    sycl::buffer buffer_a(&summand_a, sycl::range(1));
    sycl::buffer buffer_b(&summand_b, sycl::range(1));
    sycl::buffer buffer_result(&result, sycl::range(1));

    queue.submit([&](sycl::handler& handler) {
      sycl::accessor in_a{buffer_a, handler}
      sycl::accessor in_b{buffer_b, handler}
      sycl::accessor out_result{buffer_result, handler}

      handler.single_task([=]() {
        out[0] = in_a[0] + in_b[0];
      });
    }).wait();
  }
}
