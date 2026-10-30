#include "io_uring_reader.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

#include <poll.h>

namespace sglang_engram_ssd::query {

IoUringReader::IoUringReader(std::int64_t queue_depth) : queue_depth_(queue_depth) {
#if ENGRAM_SSD_ENABLE_IO_URING
  if (queue_depth <= 0 ||
      queue_depth > static_cast<std::int64_t>(std::numeric_limits<unsigned>::max())) {
    throw std::invalid_argument("Invalid io_uring queue depth");
  }
  // flags=0，不启用 SQPOLL：submit 的正返回值就是实际接收的请求数。
  const int result = io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring_, 0);
  if (result < 0) {
    throw std::system_error(-result, std::generic_category(), "io_uring_queue_init");
  }
  opened_ = true;
#else
  throw std::runtime_error(
      "io_uring backend is not built; configure ENGRAM_SSD_ENABLE_IO_URING=ON");
#endif
}

IoUringReader::~IoUringReader() {
  close();
}

void IoUringReader::read_all(const std::vector<IoReadRequest>& requests) {
#if ENGRAM_SSD_ENABLE_IO_URING
  if (!opened_) {
    throw std::runtime_error("io_uring reader is closed");
  }
  for (const auto& request : requests) {
    if (request.destination == nullptr || request.bytes <= 0 || request.offset < 0) {
      throw std::invalid_argument("Invalid io_uring read request");
    }
    if (request.bytes - 1 >
        std::numeric_limits<std::int64_t>::max() - request.offset) {
      throw std::overflow_error("io_uring read range exceeds int64 range");
    }
  }

  // 所有窗口内存都在提交 I/O 前准备好，不在有在途请求时扩容。
  std::vector<PendingRead> window;
  window.reserve(std::min(requests.size(), static_cast<std::size_t>(queue_depth_)));
  std::size_t next = 0;
  while (next < requests.size()) {
    window.clear();
    while (next < requests.size() &&
           window.size() < static_cast<std::size_t>(queue_depth_)) {
      window.push_back(PendingRead{requests[next], 0, 0});
      ++next;
    }
    read_window(window);
  }
#else
  throw std::runtime_error("io_uring backend is not built");
#endif
}

void IoUringReader::close() {
#if ENGRAM_SSD_ENABLE_IO_URING
  if (opened_) {
    // read_all 在返回或抛错前已收齐提交的 CQE；这里不把 queue_exit 当作 I/O 屏障。
    io_uring_queue_exit(&ring_);
    opened_ = false;
  }
#endif
}

#if ENGRAM_SSD_ENABLE_IO_URING
void IoUringReader::read_window(std::vector<PendingRead>& window) {
  while (true) {
    unsigned prepared = 0;
    for (std::size_t index = 0; index < window.size(); ++index) {
      auto& pending = window[index];
      if (pending.received == pending.request.bytes) {
        continue;
      }
      auto* sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) {
        // 上一轮已收齐完成事件；当前只有尚未提交的 SQE，可以安全销毁 ring。
        close();
        throw std::runtime_error("io_uring submission queue has no free entry");
      }
      pending.submitted_bytes = static_cast<unsigned>(std::min<std::int64_t>(
          pending.request.bytes - pending.received, std::numeric_limits<int>::max()));
      io_uring_prep_read(
          sqe,
          pending.request.fd,
          pending.request.destination + pending.received,
          pending.submitted_bytes,
          static_cast<std::uint64_t>(pending.request.offset + pending.received));
      io_uring_sqe_set_data64(sqe, index);
      ++prepared;
    }
    if (prepared == 0) {
      return;
    }

    // submit 可能只接收部分 SQE，必须累加实际提交数；不使用 errno 判断 liburing 错误。
    unsigned submitted = 0;
    int submit_error = 0;
    while (submitted < prepared) {
      const int result = io_uring_submit(&ring_);
      if (result == -EINTR) {
        continue;
      }
      if (result <= 0) {
        submit_error = result < 0 ? -result : EIO;
        break;
      }
      submitted += static_cast<unsigned>(result);
    }

    int wait_error = 0;
    int read_error = 0;
    bool eof = false;
    std::size_t failed_index = 0;
    for (unsigned completed = 0; completed < submitted;) {
      io_uring_cqe* cqe = nullptr;
      const int result = io_uring_wait_cqe(&ring_, &cqe);
      if (result < 0) {
        if (result != -EINTR) {
          if (wait_error == 0) {
            wait_error = -result;
          }
          // 不能带着在途写入直接抛错；继续回收完成事件，并避免错误路径忙等。
          ::poll(nullptr, 0, 1);  // 等待 1 ms；此处不能抛出异常让在途缓冲区提前失效。
        }
        continue;
      }

      const auto index = io_uring_cqe_get_data64(cqe);
      const int bytes = cqe->res;
      if (index >= window.size()) {
        if (read_error == 0) {
          read_error = EIO;
        }
      } else {
        auto& pending = window[static_cast<std::size_t>(index)];
        if (bytes > 0 && static_cast<unsigned>(bytes) <= pending.submitted_bytes) {
          pending.received += bytes;
        } else if (bytes != -EINTR && read_error == 0) {
          failed_index = static_cast<std::size_t>(index);
          eof = bytes == 0;
          read_error = bytes < 0 ? -bytes : EIO;
        }
      }
      io_uring_cqe_seen(&ring_, cqe);
      ++completed;
    }

    // 控制面失败时先收齐已接收的请求，再销毁 ring，丢弃未提交 SQE，禁止误用旧地址。
    if (submit_error != 0 || wait_error != 0) {
      close();
      const int error = submit_error != 0 ? submit_error : wait_error;
      throw std::system_error(error, std::generic_category(), "io_uring submit/wait");
    }
    if (read_error != 0) {
      const auto& pending = window[failed_index];
      if (eof) {
        throw std::runtime_error(
            "io_uring unexpected EOF: fd=" + std::to_string(pending.request.fd) +
            ", offset=" + std::to_string(pending.request.offset) +
            ", expected=" + std::to_string(pending.request.bytes) +
            ", received=" + std::to_string(pending.received));
      }
      throw std::system_error(
          read_error,
          std::generic_category(),
          "io_uring read: fd=" + std::to_string(pending.request.fd) +
              ", offset=" + std::to_string(pending.request.offset));
    }
    // 本轮已经全部结束；下一轮只重提短读和 EINTR 的剩余部分。
  }
}
#endif

}  // namespace sglang_engram_ssd::query
