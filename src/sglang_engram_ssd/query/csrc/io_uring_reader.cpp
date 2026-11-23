// io_uring 滑动窗口提交与完成回收；声明在 io_uring_reader.h，经 bindings.cpp 的 reader 绑定调用。
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
}

IoUringReader::~IoUringReader() {
  close();
}

void IoUringReader::read_all(const std::vector<IoReadRequest>& requests) {
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

  // 进度表在提交前分配完毕；CQE 的 user_data 使用稳定的输入下标，不受补交次序影响。
  // 此后直到收齐在途请求都不分配内存、不构造异常，避免提前退出后内核继续写调用方缓冲区。
  std::vector<PendingRead> pending(requests.size());
  std::vector<std::size_t> refill(std::min(requests.size(), static_cast<std::size_t>(queue_depth_)));
  std::size_t refill_count = 0;
  std::size_t next = 0;
  unsigned in_flight = 0;
  unsigned prepared = 0;
  int control_error = 0;
  int read_error = 0;
  bool eof = false;
  std::size_t failed_index = 0;

  // 输入 index 为 requests／pending 的稳定下标；只准备剩余字节，不提交内核。
  // 成功时写 pending[index].submitted_bytes、填充 SQE 并递增 prepared；无空 SQE 时仅记录 control_error。
  // 捕获引用由本次 read_all 持有，全部 I/O 收齐前不能退出其作用域。
  const auto prepare = [&](std::size_t index) {
    auto* sqe = io_uring_get_sqe(&ring_);
    if (sqe == nullptr) {
      control_error = EIO;
      return;
    }
    const auto& request = requests[index];
    auto& progress = pending[index];
    progress.submitted_bytes = static_cast<unsigned>(std::min<std::int64_t>(
        request.bytes - progress.received, std::numeric_limits<int>::max()));
    io_uring_prep_read(
        sqe, request.fd, request.destination + progress.received, progress.submitted_bytes,
        static_cast<std::uint64_t>(request.offset + progress.received));
    io_uring_sqe_set_data64(sqe, index);
    ++prepared;
  };

  // flags=0 且 ring 由当前调用独占；只把 submit 正返回值计入在途数。
  // 部分提交后失败时，尚未接收的 SQE 留在 ring 中，收齐已接收请求后关闭 ring 丢弃它们。
  // 输入为捕获的 prepared 个 SQE；无返回值，扣减 prepared、累加 in_flight，失败记录 control_error。
  const auto submit_prepared = [&]() {
    unsigned submitted = 0;
    while (submitted < prepared) {
      const int result = io_uring_submit(&ring_);
      if (result == -EINTR) {
        continue;
      }
      if (result <= 0) {
        control_error = result < 0 ? -result : EIO;
        break;
      }
      submitted += static_cast<unsigned>(result);
      in_flight += static_cast<unsigned>(result);
    }
    prepared -= submitted;
  };

  while (next < requests.size() && prepared < static_cast<unsigned>(queue_depth_)) {
    prepare(next++);
    if (control_error != 0) {
      break;
    }
  }
  if (control_error == 0) {
    submit_prepared();
  }

  while (in_flight > 0) {
    io_uring_cqe* cqe = nullptr;
    const int result = io_uring_wait_cqe(&ring_, &cqe);
    if (result < 0) {
      if (result != -EINTR) {
        if (control_error == 0) {
          control_error = -result;
        }
        // 等待失败也不能释放仍在写的内存；停止补交，休眠后继续收齐完成事件。
        ::poll(nullptr, 0, 1);
      }
      continue;
    }

    const auto index = io_uring_cqe_get_data64(cqe);
    const int bytes = cqe->res;
    if (index >= requests.size()) {
      if (read_error == 0) {
        read_error = EIO;
      }
    } else {
      auto& progress = pending[static_cast<std::size_t>(index)];
      if (bytes > 0 && static_cast<unsigned>(bytes) <= progress.submitted_bytes) {
        progress.received += bytes;
      } else if (bytes != -EINTR && read_error == 0) {
        failed_index = static_cast<std::size_t>(index);
        eof = bytes == 0;
        read_error = bytes < 0 ? -bytes : EIO;
      }
    }
    io_uring_cqe_seen(&ring_, cqe);
    --in_flight;

    if (control_error == 0 && read_error == 0) {
      // 短读与 EINTR 优先重提同一请求；完整读完才腾出位置给新请求。
      // 每个 CQE 至多产生一个补交项，所以在途数加待补交数始终不超过 queue_depth。
      if (pending[index].received < requests[index].bytes) {
        refill[refill_count++] = static_cast<std::size_t>(index);
      } else if (next < requests.size()) {
        refill[refill_count++] = next++;
      }
    }

    // 只顺带收回已经到达的 CQE，再一起补交以减少 submit 系统调用；不等待同批最慢的请求。
    // 待补交列表尚未变成 SQE，若本批 CQE 中发现读错可直接丢弃，ring 仍能安全复用。
    if (in_flight > 0 && io_uring_cq_ready(&ring_) > 0) {
      continue;
    }
    if (control_error != 0 || read_error != 0) {
      continue;
    }
    for (std::size_t i = 0; i < refill_count; ++i) {
      prepare(refill[i]);
      if (control_error != 0) {
        break;
      }
    }
    refill_count = 0;
    if (control_error == 0 && prepared > 0) {
      submit_prepared();
    }
  }

  // 此时没有在途写入，才允许构造并抛出异常。控制面错误关闭 ring；读错误后仍可再次读取。
  if (control_error != 0) {
    close();
    throw std::system_error(control_error, std::generic_category(), "io_uring submit/wait");
  }
  if (read_error != 0) {
    const auto& request = requests[failed_index];
    if (eof) {
      throw std::runtime_error(
          "io_uring unexpected EOF: fd=" + std::to_string(request.fd) +
          ", offset=" + std::to_string(request.offset) +
          ", expected=" + std::to_string(request.bytes) +
          ", received=" + std::to_string(pending[failed_index].received));
    }
    throw std::system_error(
        read_error, std::generic_category(),
        "io_uring read: fd=" + std::to_string(request.fd) +
            ", offset=" + std::to_string(request.offset));
  }
}

void IoUringReader::close() {
  if (opened_) {
    // read_all 在返回或抛错前已收齐提交的 CQE；这里不把 queue_exit 当作 I/O 屏障。
    io_uring_queue_exit(&ring_);
    opened_ = false;
  }
}

}  // namespace sglang_engram_ssd::query
