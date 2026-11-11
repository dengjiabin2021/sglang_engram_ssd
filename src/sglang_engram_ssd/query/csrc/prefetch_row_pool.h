// pinned 预取行池与引用句柄；实现在 prefetch_row_pool.cpp，Python 绑定在 bindings.cpp。
#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sglang_engram_ssd::query {

class PrefetchRowPool;

class PrefetchRowHandle {
 public:
  PrefetchRowHandle(const PrefetchRowHandle&) = delete;
  PrefetchRowHandle& operator=(const PrefetchRowHandle&) = delete;

  // 返回 CPU contiguous int64 [M]，逐项对应本句柄新建的 PENDING 行和槽位。
  // 句柄持有池的生命周期；数组只供读取，不得修改或在 release/cancel 后用于 I/O。
  // 调用方必须等所有相关 I/O 完成后再 release/cancel；不会自动释放引用。
  at::Tensor missing_row_ids() const;
  at::Tensor missing_slot_ids() const;

 private:
  friend class PrefetchRowPool;
  explicit PrefetchRowHandle(std::shared_ptr<PrefetchRowPool> owner);

  std::shared_ptr<PrefetchRowPool> owner_;
  std::vector<std::int64_t> slots_;
  std::vector<std::int64_t> created_slots_;
  at::Tensor missing_row_ids_;
  at::Tensor missing_slot_ids_;
  bool released_ = false;
};

struct PrefetchRowResolution {
  at::Tensor slot_ids;  // CPU contiguous int64 [Q]，未命中为 -1。
  at::Tensor hit_mask;  // CPU contiguous bool [Q]，只有 READY 才命中。
  std::shared_ptr<PrefetchRowHandle> handle;  // 每个唯一命中槽增加一次临时引用。
};

struct PrefetchRowPoolStats {
  std::int64_t capacity = 0;
  std::int64_t payload_bytes = 0;
  std::int64_t occupied_slots = 0;
  std::int64_t referenced_slots = 0;
  std::int64_t pending_slots = 0;
  std::int64_t ready_slots = 0;
  std::int64_t cached_slots = 0;
  std::int64_t submitted_rows = 0;
  std::int64_t dedup_saved_rows = 0;
  std::int64_t forward_hits = 0;
  std::int64_t forward_misses = 0;
  double miss_read_ms = 0;
  std::int64_t deferred_requests = 0;
  bool tracking_enabled = false;
  bool closed = false;
};

class PrefetchRowPool : public std::enable_shared_from_this<PrefetchRowPool> {
 public:
  // 同步分配当前 CUDA 设备可访问的两张 CPU uint8 pinned 表，失败直接抛出异常。
  // 容量 S=pool_bytes/(D+D/32)，两个表起点均 4096 对齐且连续，保留原分配的 Tensor。
  // pool_bytes 只限制量化 payload；对齐余量、槽位元数据和索引不计入该值。
  // 实例须由 shared_ptr 管理，返回的句柄会保留池的生命周期。
  PrefetchRowPool(std::int64_t dim, std::int64_t pool_bytes, bool track_stats = false);
  ~PrefetchRowPool() = default;
  PrefetchRowPool(const PrefetchRowPool&) = delete;
  PrefetchRowPool& operator=(const PrefetchRowPool&) = delete;

  // 返回共享存储的 CPU contiguous uint8 [S,D]、[S,D/32] Tensor 与对应设备地址。
  // 调用方只能写 acquire 返回的新 PENDING 槽；读 GPU 期间须保留 resolve 的句柄。
  at::Tensor weight() const;
  at::Tensor scale() const;
  std::uintptr_t weight_device_ptr() const;
  std::uintptr_t scale_device_ptr() const;
  std::int64_t capacity() const;
  std::int64_t weight_stride() const;
  std::int64_t scale_stride() const;

  // CPU contiguous int64 [Q]，保存 layer_id 的全局行号；不执行 I/O。
  // 按唯一行增加引用，原有 PENDING/READY 复用同一槽；新行返回在 handle.missing_* 中。
  // 空间不足返回 nullptr，引用和索引不变；失败 key 显式抛错，不会永久等待。
  std::shared_ptr<PrefetchRowHandle> acquire(std::int64_t layer_id, const at::Tensor& row_ids);

  // 同步检查句柄的所有槽是否 READY；空句柄为 true，已释放/失败句柄抛出异常。
  bool is_ready(const std::shared_ptr<PrefetchRowHandle>& handle) const;

  // slot_ids 为 CPU contiguous int64 [M]；必须等同步读取返回后再发布 READY。
  // 只允许有效 PENDING 槽，整批先检查再更新；重复槽位只处理一次。
  void mark_ready(const at::Tensor& slot_ids);

  // CPU contiguous int64 [Q]，返回原顺序/重复位置的槽号、命中掩码和临时引用句柄。
  // 只命中 READY，PENDING 为 miss；失败 key 抛错。GPU gather 完成后必须 release。
  PrefetchRowResolution resolve(std::int64_t layer_id, const at::Tensor& row_ids);

  // 每个句柄恰好释放一次引用，重复调用幂等；不撤销 PENDING 的 I/O。
  // PENDING 即使引用归零也保留到 mark_ready，防止覆盖仍被 I/O 使用的槽位。
  void release(const std::shared_ptr<PrefetchRowHandle>& handle);

  // 仅在 I/O 未开始或失败且所有 CQE 已收齐后调用；同时释放本句柄的全部引用。
  // 只撤销本句柄新建的 PENDING：独占则回收，共享则留失败标记直到最后引用释放。
  // 其他句柄的 PENDING 和所有 READY 只减引用，不删除共享数据；重复调用幂等。
  void cancel(const std::shared_ptr<PrefetchRowHandle>& handle);

  // 在锁下取得原生快照；占用状态总可读，累计统计只在 track_stats=true 时更新。
  PrefetchRowPoolStats stats() const;

  // 同步记录非负行数与有限非负毫秒数；关闭统计时直接忽略，不执行 I/O。
  void record_forward(std::int64_t hits, std::int64_t misses, double miss_ms);
  // 同步记录非负暂缓请求数；关闭统计时直接忽略，不执行 I/O。
  void record_deferred(std::int64_t count);

  // 调用方先停提交并收齐 I/O/GPU 引用。存在引用或 PENDING 时拒绝关闭；成功后幂等。
  // 不打开或关闭模型 fd；释放池自身的 Tensor、元数据和索引。
  void close();

 private:
  enum class State : std::uint8_t { FREE, PENDING, READY };
  struct RowKey {
    std::int64_t layer_id = -1;
    std::int64_t row_id = -1;
    bool operator==(const RowKey&) const = default;
  };
  struct RowKeyHash {
    std::size_t operator()(const RowKey& key) const;
  };
  struct Slot {
    RowKey key;
    std::int64_t references = 0;
    std::int64_t previous = -1;
    std::int64_t next = -1;
    State state = State::FREE;
    bool failed = false;
  };

  // 以下辅助函数仅在 mutex_ 已持有时调用，不做 I/O 或访问 Python 对象。
  void require_open() const;
  void require_handle(const std::shared_ptr<PrefetchRowHandle>& handle) const;
  void require_healthy(const Slot& slot) const;
  void remove_lru(std::int64_t slot_id);
  void add_lru(std::int64_t slot_id);
  void pin(std::int64_t slot_id);
  void unpin(std::int64_t slot_id);
  void recycle(std::int64_t slot_id);
  std::int64_t take_slot();
  void release_slots(PrefetchRowHandle& handle);

  const std::int64_t capacity_;
  const std::int64_t weight_stride_;
  const std::int64_t scale_stride_;
  const bool track_stats_;
  at::Tensor weight_storage_;
  at::Tensor scale_storage_;
  at::Tensor weight_;
  at::Tensor scale_;
  std::uintptr_t weight_device_ptr_ = 0;
  std::uintptr_t scale_device_ptr_ = 0;
  mutable std::mutex mutex_;
  std::vector<Slot> slots_;
  std::unordered_map<RowKey, std::int64_t, RowKeyHash> index_;
  std::int64_t free_head_ = -1;
  std::int64_t lru_head_ = -1;
  std::int64_t lru_tail_ = -1;
  std::int64_t free_slots_ = 0;
  std::int64_t referenced_slots_ = 0;
  std::int64_t pending_slots_ = 0;
  std::int64_t ready_slots_ = 0;
  std::int64_t cached_slots_ = 0;
  PrefetchRowPoolStats counters_;
  bool closed_ = false;
};

}  // namespace sglang_engram_ssd::query
