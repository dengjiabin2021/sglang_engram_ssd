// 按行号从 checkpoint 读取 Engram 行的原始字节（FP8 weight 与 E8M0 scale）。
// 实现在 batch_reader.cpp，Python 绑定在 bindings.cpp，底层 I/O 由 IoUringReader 完成。
#pragma once

#include "io_uring_reader.h"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sglang_engram_ssd::query {

// 单层 Engram 表在 checkpoint 中的位置。字段与 query/interface.py 的 RowTableSpec 一一对应。
// 第 r 行的 weight 位于 weight_fd 的 [weight_offset + r*dim, +dim)，
// scale 位于 scale_fd 的 [scale_offset + r*(dim/32), +dim/32)。两者可以在不同文件中。
// runtime 创建，reader 复制后只读使用；不拥有 fd，跨线程只读无需锁。
struct RowTableSpec {
  std::int64_t layer_id;
  std::int64_t rows;            // 全局行数，合法行号为 [0, rows)。
  std::int64_t dim;             // weight 每行字节数；scale 每行 dim/32 字节。
  int weight_fd;                // 借用 runtime 打开的只读 fd，reader 不关闭。
  int scale_fd;
  std::int64_t weight_offset;   // 张量数据在文件中的绝对起始字节。
  std::int64_t scale_offset;
};

// 同步读取器：每次调用都等全部数据读完才返回，失败时抛异常。
// 一个实例只有一个 io_uring，不是线程安全的：同一实例的调用必须串行（Python 包装用锁保证）。
// 不做缓存、不去重、不做 CUDA 操作；重复行号会被重复读取。
// runtime 创建并在全部调用结束后关闭；表元数据由 reader 复制，工作区由调用方独立持有。
class BatchRowReader {
 public:
  // 复制表元数据并创建 io_uring。queue depth 必须为正，初始化失败抛异常。
  // tables 可以为空，用于模型加载前单独检查 io_uring 是否可用。
  // 借用的 fd 必须在 reader 关闭之后才能关闭。
  BatchRowReader(
      const std::vector<RowTableSpec>& tables,
      std::int64_t io_uring_queue_depth = 32);
  // 无输入输出；销毁时同步释放 ring，与 close 等效，调用方须先结束读取。
  ~BatchRowReader();

  BatchRowReader(const BatchRowReader&) = delete;
  BatchRowReader& operator=(const BatchRowReader&) = delete;

  // 分配一对 CPU pinned uint8 工作区 [capacity,D] 和 [capacity,D/32]，由调用方持有并复用。
  // 行宽相同的层可以共用同一对工作区。分配失败抛异常，不回退到普通内存。
  std::pair<at::Tensor, at::Tensor> allocate_workspace(
      std::int64_t layer_id, std::int64_t capacity) const;

  // 按顺序读取：第 i 个行号写到 weight_out[i] 和 scale_out[i]。
  //   row_ids:   CPU int64 [Q]，contiguous，全局行号，可重复。
  //   weight_out: CPU uint8 [K,D]，scale_out: CPU uint8 [K,D/32]，contiguous，要求 Q <= K。
  // 只写前 Q 行，其余行不变。输出与输入的内存不得重叠。
  // 返回时所有数据已写完，函数不保留任何张量指针。
  void read_rows(
      std::int64_t layer_id,
      const at::Tensor& row_ids,
      at::Tensor& weight_out,
      at::Tensor& scale_out);

  // 按槽号写入：第 i 个行号写到 weight_base[slot_ids[i]] 和 scale_base[slot_ids[i]]。
  // 供预取行池使用，weight_base/scale_base 即池的两张表（CPU uint8 [S,D]、[S,D/32]）。
  //   row_ids、slot_ids: 等长 CPU int64 [Q]，contiguous；slot_ids 必须互不重复且在 [0,S) 内。
  // 只写这些槽，不修改池的状态：调用方在返回后自行 mark_ready。
  // 出错时也会先等所有已提交的读取结束再抛异常，保证返回后不再有写入。
  void read_rows_to_slots(
      std::int64_t layer_id,
      const at::Tensor& row_ids,
      const at::Tensor& slot_ids,
      at::Tensor& weight_base,
      at::Tensor& scale_base);

  // 释放 io_uring 和表元数据；不关闭借用的 fd。调用前应确保没有读取在进行。
  void close();

 private:
  // 输入 layer_id；返回借用的表元数据引用，生命周期到 close，reader 关闭／层不存在时抛异常。
  const RowTableSpec& open_table(std::int64_t layer_id) const;

  // 输入为 table 和待读取的 row_ids／两个输出，约束与 read_rows 相同。
  // 无返回值或状态修改；非法设备、dtype、形状、连续性或容量时抛异常。
  void check_inputs(
      const RowTableSpec& table,
      const at::Tensor& row_ids,
      const at::Tensor& weight_out,
      const at::Tensor& scale_out) const;

  // 输入非负 tensor_offset、row_id 和正数 row_bytes，单位为字节／全局行号。
  // 返回该行绝对字节偏移；溢出 off_t 范围抛异常，不修改元数据。
  std::int64_t row_offset(
      std::int64_t tensor_offset,
      std::int64_t row_id,
      std::int64_t row_bytes) const;

  // read_rows 与 read_rows_to_slots 的共同实现：每行生成 weight、scale 两个读请求，交给 io_uring。
  // slot_ids 为 nullptr 时第 i 行写到第 i 个输出位置，否则写到 slot_ids[i]。
  // weight_stride/scale_stride 是输出中相邻两行的字节距离。
  // table 仅借用；row_ids/slot_ids 是 CPU int64 [query_count] 的地址，weights/scales 是输出基址。
  // 无返回值，直接写输出；同步收齐 I/O 后返回或抛异常，不保存任何输入指针。
  void read_rows_uring(
      const RowTableSpec& table,
      const std::int64_t* row_ids,
      const std::int64_t* slot_ids,
      std::int64_t query_count,
      std::uint8_t* weights,
      std::uint8_t* scales,
      std::int64_t weight_stride,
      std::int64_t scale_stride);

  std::unordered_map<std::int64_t, RowTableSpec> tables_;  // 层号 -> 表位置。
  std::unique_ptr<IoUringReader> io_uring_reader_;
  bool closed_ = false;
};

}  // namespace sglang_engram_ssd::query
