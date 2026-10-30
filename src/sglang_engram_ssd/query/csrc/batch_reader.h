// 批量行查询类定义；同步读取在 batch_reader.cpp，Python 绑定在 bindings.cpp。
#pragma once

#include "thread_pool.h"
#include "io_uring_reader.h"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <string>
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
  // cache_bytes 是当前 reader 所有层共用的行数据预算，0 表示不保留缓存。
  // pread 下 num_threads 缺省为 0，正数启用线程池；其等待行任务容量独立配置。
  // io_uring 只使用自己的在途文件请求容量；两种队列容量缺省均为 32。
  // nullopt 表示未指定；显式传入当前后端不使用的参数时仅 warning，不校验其值。
  BatchRowReader(
      const std::vector<RowTableSpec>& tables,
      std::int64_t cache_bytes,
      std::optional<std::int64_t> num_threads = std::nullopt,
      std::optional<std::int64_t> thread_pool_queue_depth = std::nullopt,
      std::optional<std::int64_t> io_uring_queue_depth = std::nullopt,
      const std::string& io_backend = "pread");
  ~BatchRowReader();

  BatchRowReader(const BatchRowReader&) = delete;
  BatchRowReader& operator=(const BatchRowReader&) = delete;

  // 一次性分配 [capacity,D]、[capacity,D/32] 的 CPU uint8 pinned 工作区。
  // 返回的 Tensor 持有内存，调用方保存并复用，相同行宽的层可共用。
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

  // 将一行读满到目标内存，处理短读、信号中断和文件错误。
  void read_exact_at(
      int fd,
      std::uint8_t* destination,
      std::int64_t bytes,
      std::int64_t offset) const;

  // 只读取一行文件数据，不操作缓存；串行路径和 worker 共用。
  void read_file_row(
      const RowTableSpec& table,
      std::int64_t row_id,
      std::uint8_t* weight,
      std::uint8_t* scale) const;

  // 调用线程负责缓存，worker 仅读文件；等待全部任务后按查询顺序更新 LRU。
  void read_rows_threaded(
      const RowTableSpec& table,
      const std::int64_t* row_ids,
      std::int64_t query_count,
      std::uint8_t* weights,
      std::uint8_t* scales);

  // 将未命中的 weight/scale 组成异步请求，完成后仍按原查询顺序更新缓存。
  void read_rows_uring(
      const RowTableSpec& table,
      const std::int64_t* row_ids,
      std::int64_t query_count,
      std::uint8_t* weights,
      std::uint8_t* scales);

  void commit_cache_rows(
      const RowTableSpec& table,
      const std::int64_t* row_ids,
      std::int64_t query_count,
      const std::uint8_t* weights,
      const std::uint8_t* scales);

  // 返回缓存字节；update_lru=false 用于并行提交阶段，只查找、不移动节点。
  const std::vector<std::uint8_t>* find_cached_row(
      std::int64_t layer_id, std::int64_t row_id, bool update_lru = true);

  // 只在两段数据均读取成功后调用；空间不足时先淘汰最久未使用的行。
  void cache_row(
      std::int64_t layer_id,
      std::int64_t row_id,
      const std::uint8_t* weight,
      const std::uint8_t* scale);

  // 删除全局 LRU 队尾的一行，并同步更新索引和字节计数。
  void evict_oldest_row();

  struct CachedRow {
    std::int64_t layer_id;
    std::int64_t row_id;
    std::vector<std::uint8_t> data;  // 前 D 字节为 weight，后 D/32 字节为 scale。
  };
  using CacheIterator = std::list<CachedRow>::iterator;
  using RowIndex = std::unordered_map<std::int64_t, CacheIterator>;

  // 表元数据与行缓存分开管理；fd 仍由 runtime 持有。
  std::unordered_map<std::int64_t, RowTableSpec> tables_;
  std::int64_t cache_budget_bytes_;
  std::int64_t cached_bytes_ = 0;  // 只统计缓存 payload，不含容器和索引开销。
  std::list<CachedRow> cache_lru_; // 队头最近使用，队尾最久未使用，所有层共用。
  std::unordered_map<std::int64_t, RowIndex> cache_index_;  // 层号 -> 行号 -> 链表位置。
  std::unique_ptr<ThreadPool> thread_pool_;  // 仅 pread 的多线程模式创建。
  std::unique_ptr<IoUringReader> io_uring_reader_;  // 与线程池互斥，只在显式选择时创建。
  bool closed_ = false;
};

}  // namespace sglang_engram_ssd::query
