// 固定线程数、有界等待队列的同步读取线程池。
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace sglang_engram_ssd::query {

class ThreadPool {
 public:
  ThreadPool(std::int64_t num_threads, std::int64_t queue_depth);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  // 队列满时等待空位；future 保存任务完成状态和读取异常。
  std::future<void> submit(std::function<void()> task);

  // 停止接收新任务，排空已提交任务并等待全部 worker 退出，可重复调用。
  void shutdown();

 private:
  void worker_loop();

  std::size_t queue_depth_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  bool stopping_ = false;
};

}  // namespace sglang_engram_ssd::query
