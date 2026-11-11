// 批量行查询类定义；同步读取在 batch_reader.cpp，Python 绑定在 bindings.cpp。
#pragma once

#include "io_uring_reader.h"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sglang_engram_ssd::query {

// 字段与 query/interface.py 的 RowTableSpec 一一对应。
struct RowTableSpec {
  std::int64_t layer_id;
  std::int64_t rows;
  std::int64_t dim;
  int weight_fd;                // 借用 runtime 的 fd，不转移所有权。
  int scale_fd;
  std::int64_t weight_offset;   // 原文件中的绝对字节偏移。
  std::int64_t scale_offset;
};

class BatchRowReader {
 public:
  // 同步创建 io_uring；queue depth 必须为正，初始化失败直接抛出异常。
  // tables 只复制元数据并借用 fd；允许空列表用于模型加载前的初始化检查。
  // 同一实例的读取和关闭由调用方串行执行，fd 必须存活到 reader 关闭之后。
  BatchRowReader(
      const std::vector<RowTableSpec>& tables,
      std::int64_t io_uring_queue_depth = 32);
  ~BatchRowReader();

  BatchRowReader(const BatchRowReader&) = delete;
  BatchRowReader& operator=(const BatchRowReader&) = delete;

  // 一次性分配 [capacity,D]、[capacity,D/32] 的 CPU uint8 pinned 工作区。
  // 返回的 Tensor 持有内存，调用方保存并复用，相同行宽的层可共用；失败时抛出异常。
  std::pair<at::Tensor, at::Tensor> allocate_workspace(
      std::int64_t layer_id, std::int64_t capacity) const;

  // row_ids 为连续 CPU int64 [Q]，两个输出为连续 CPU uint8 [K,D]、[K,D/32]。
  // Q <= K，仅填充前 Q 行；保留输入顺序和重复行，输出原始字节。
  // 同步读满后返回，不保留 Tensor 引用或指针供返回后访问；失败时抛出异常。
  // 输出可直接传入 allocate_workspace 返回的 Tensor，不创建中间输出张量。
  // 输入与两个输出的内存区间不得重叠。
  // 不执行 D2H、H2D、反量化或 TP 通信，调用期间不得并发复用输入和输出。
  void read_rows(
      std::int64_t layer_id,
      const at::Tensor& row_ids,
      at::Tensor& weight_out,
      at::Tensor& scale_out);

  // row_ids/slot_ids 为等长 CPU contiguous int64 [Q]，slot_ids 必须互不重复。
  // weight_base/scale_base 为 CPU contiguous uint8 [S,D]、[S,D/32]；只写指定槽位。
  // 同步读满全部 weight/scale 后返回，失败也先回收所有在途请求；不发布池状态。
  // 调用方保留 Tensor 与 PENDING 槽引用，返回后再 mark_ready；GPU 不能同时读这些槽。
  // fd 只借用；不执行 CUDA 操作、不建立缓存，非法行号/槽号和 I/O 错误均抛出异常。
  void read_rows_to_slots(
      std::int64_t layer_id,
      const at::Tensor& row_ids,
      const at::Tensor& slot_ids,
      at::Tensor& weight_base,
      at::Tensor& scale_base);

  // 在所有读取结束后调用；释放自身资源，但不关闭外部传入的 fd。
  void close();

 private:
  // 按层号取得表信息，同时检查 reader 是否已关闭。
  const RowTableSpec& open_table(std::int64_t layer_id) const;

  // 检查 CPU 张量的类型、形状、连续性和输出容量。
  void check_inputs(
      const RowTableSpec& table,
      const at::Tensor& row_ids,
      const at::Tensor& weight_out,
      const at::Tensor& scale_out) const;

  // 根据全局行号计算该行在原文件中的起点，并检查字节范围。
  std::int64_t row_offset(
      std::int64_t tensor_offset,
      std::int64_t row_id,
      std::int64_t row_bytes) const;

  // 每个查询槽分别读取 weight/scale，全部请求完成后才返回。
  void read_rows_uring(
      const RowTableSpec& table,
      const std::int64_t* row_ids,
      const std::int64_t* slot_ids,
      std::int64_t query_count,
      std::uint8_t* weights,
      std::uint8_t* scales,
      std::int64_t weight_stride,
      std::int64_t scale_stride);

  // 只保存表元数据；fd 仍由 runtime 持有。
  std::unordered_map<std::int64_t, RowTableSpec> tables_;
  std::unique_ptr<IoUringReader> io_uring_reader_;
  bool closed_ = false;
};

}  // namespace sglang_engram_ssd::query
