#include "thread_pool.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace sglang_engram_ssd::query {

ThreadPool::ThreadPool(std::int64_t num_threads, std::int64_t queue_depth)
    : queue_depth_(static_cast<std::size_t>(queue_depth)) {
  if (num_threads <= 0 || queue_depth <= 0) {
    throw std::invalid_argument("Thread count and queue depth must be positive");
  }

  workers_.reserve(static_cast<std::size_t>(num_threads));
  try {
    for (std::int64_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
  } catch (...) {
    // 创建部分线程后失败，也要先唤醒并回收这些线程，再让构造失败。
    shutdown();
    throw;
  }
}

ThreadPool::~ThreadPool() {
  shutdown();
}

std::future<void> ThreadPool::submit(std::function<void()> task) {
  // packaged_task 将读取异常保存到 future，不让异常退出 worker 线程。
  auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto result = packaged->get_future();

  std::unique_lock<std::mutex> lock(mutex_);
  not_full_.wait(lock, [this] {
    return stopping_ || tasks_.size() < queue_depth_;
  });
  if (stopping_) {
    throw std::runtime_error("Cannot submit to a stopped thread pool");
  }
  tasks_.emplace([packaged] { (*packaged)(); });
  lock.unlock();
  not_empty_.notify_one();
  return result;
}

void ThreadPool::worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      not_empty_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
      if (stopping_ && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    not_full_.notify_one();
    // 执行期间不持有队列锁，各 worker 可同时进行 pread。
    task();
  }
}

void ThreadPool::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  not_empty_.notify_all();
  not_full_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

}  // namespace sglang_engram_ssd::query
