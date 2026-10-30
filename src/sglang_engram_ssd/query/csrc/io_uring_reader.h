// 可选的有界 io_uring 读取器；类定义在头文件，实现在 cpp。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#if ENGRAM_SSD_ENABLE_IO_URING
#include <liburing.h>
#endif

namespace sglang_engram_ssd::query {

// 一项请求只读取 weight 或 scale 的一段数据，fd 和输出内存均由调用方持有。
struct IoReadRequest {
  int fd;
  std::uint8_t* destination;
  std::int64_t bytes;
  std::int64_t offset;
};

class IoUringReader {
 public:
  explicit IoUringReader(std::int64_t queue_depth);
  ~IoUringReader();

  IoUringReader(const IoUringReader&) = delete;
  IoUringReader& operator=(const IoUringReader&) = delete;

  // 内部异步、外部同步：读满全部请求后才返回，失败也先回收已提交请求。
  void read_all(const std::vector<IoReadRequest>& requests);
  void close();

 private:
#if ENGRAM_SSD_ENABLE_IO_URING
  struct PendingRead {
    IoReadRequest request;
    std::int64_t received = 0;
    unsigned submitted_bytes = 0;
  };

  // 每轮提交后收齐该轮 CQE，再重提短读或 EINTR 的剩余部分。
  void read_window(std::vector<PendingRead>& window);
  io_uring ring_{};
#endif

  std::int64_t queue_depth_;
  bool opened_ = false;
};

}  // namespace sglang_engram_ssd::query
