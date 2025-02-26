#include <sycl/sycl.hpp>

int main() {
  const size_t kDataSize{1024};
  sycl::queue queue;
  double* summand_a{sycl::malloc_shared<double>(kDataSize, queue)};
  double* summand_b{sycl::malloc_shared<double>(kDataSize, queue)};
  double* result{sycl::malloc_shared<double>(kDataSize, queue)};
  auto initialization_task = queue.parallel_for(kDataSize, [=](sycl::id<1> index){
    summand_a[index] = static_cast<double>(index);
    summand_b[index] = static_cast<double>(index);
    result[index] = 0.0f;
  });

  queue.submit([&](sycl::handler& handler){
    handler.depends_on(initialization_task);
    handler.parallel_for(kDataSize, [=](sycl::id<1> index){
      result[index] = summand_a[index] + summand_b[index];
    });
  }).wait();
  sycl::free(summand_a, queue);
  sycl::free(summand_b, queue);
  sycl::free(result, queue);
}