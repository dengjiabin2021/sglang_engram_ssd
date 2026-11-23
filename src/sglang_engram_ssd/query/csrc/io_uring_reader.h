// 基于 io_uring 的批量同步读取；实现在 io_uring_reader.cpp，经 bindings.cpp 的 reader 绑定调用。
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <liburing.h>

namespace sglang_engram_ssd::query {

// 一个读请求：从 fd 的 offset 处读 bytes 字节到 destination。fd 和目标内存都由调用方持有。
// BatchRowReader 创建请求列表，read_all 借用至返回；同一目标区间不得被其他调用并发写入。
struct IoReadRequest {
  int fd;  // 只读描述符，IoUringReader 不关闭。
  std::uint8_t* destination;  // CPU 可写缓冲区，调用方保留至所有 CQE 收齐。
  std::int64_t bytes;  // 需要读满的正数字节数，短读时继续读取剩余部分。
  std::int64_t offset;  // 非负绝对文件字节偏移，不依赖共享 fd 的当前位置。
};

// 对外是同步接口：read_all 读完全部请求才返回。内部最多保持 queue_depth 个在途请求：
//   每收回一个 CQE，安排重提该请求的剩余部分，或补入下一个请求，不等待整批完成。
//   已到达的 CQE 可合批收回与补交，减少 submit 系统调用；不会等待尚未完成的同批请求。
//   一旦发现错误便停止补交，收齐已提交请求后再抛出，保证返回后不再写目标内存。
// 一个实例只有一个 ring，不是线程安全的，调用方须串行使用。
// BatchRowReader 创建并持有，销毁前须等所有调用返回；不持有目标内存和模型 fd 的所有权。
class IoUringReader {
 public:
  // 同步创建独立 ring；queue_depth 为正数，初始化失败抛异常；不打开或持有任何 fd。
  explicit IoUringReader(std::int64_t queue_depth);
  // 无输入输出；销毁时释放 ring，调用方须先等 read_all 返回。
  ~IoUringReader();

  IoUringReader(const IoUringReader&) = delete;
  IoUringReader& operator=(const IoUringReader&) = delete;

  // 输入 requests 为 CPU 缓冲区的读取描述列表，可为空；目标地址、fd 必须存活到返回。
  // 无返回值，读满每个 destination 后返回；出错先收齐已提交请求，不留在途写入，再抛异常。
  // submit/wait 本身失败时会关闭 ring，之后该实例不可再用。
  void read_all(const std::vector<IoReadRequest>& requests);
  // 调用方必须先等 read_all 返回；同步释放 ring，幂等，不关闭借用的 fd。
  void close();

 private:
  // 单次 read_all 持有的输入进度，稳定对应一个请求下标；仅该调用访问，无需额外锁。
  // 在提交前一次性分配，避免在途期间扩容失败而提前退出；不持有模型 fd 或缓冲区。
  struct PendingRead {
    std::int64_t received = 0;      // 已读到的字节数。
    unsigned submitted_bytes = 0;   // 本轮提交的字节数，用于校验 CQE 返回值。
  };

  io_uring ring_{};  // 本实例独占，只有调用方串行访问。

  std::int64_t queue_depth_;  // 单 ring 同时在途的请求上限。
  bool opened_ = false;  // queue_init 成功后为 true，close 后不再接受请求。
};

}  // namespace sglang_engram_ssd::query
