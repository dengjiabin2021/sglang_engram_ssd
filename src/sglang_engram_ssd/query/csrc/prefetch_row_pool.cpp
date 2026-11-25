// pinned 行池的槽状态、引用与 LRU 实现；声明在 prefetch_row_pool.h，Python 绑定在 bindings.cpp。
#include "prefetch_row_pool.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/zeros.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace sglang_engram_ssd::query {
namespace {

constexpr std::int64_t kAlignment = 4096;

// 输入 dim 为正数行宽 D 且为 32 的倍数，pool_bytes 为至少容纳一行的 payload 字节预算。
// 返回槽位数 S；校验失败抛异常，不分配内存，也不修改池状态。
std::int64_t pool_capacity(std::int64_t dim, std::int64_t pool_bytes) {
  if (dim <= 0 || dim % 32 != 0 || dim > std::numeric_limits<std::int64_t>::max() - dim / 32) {
    throw std::invalid_argument("Engram prefetch dim must be a positive multiple of 32");
  }
  if (pool_bytes <= 0 || pool_bytes / (dim + dim / 32) == 0) {
    throw std::invalid_argument("Engram prefetch pool_bytes must hold at least one row");
  }
  return pool_bytes / (dim + dim / 32);
}

// 输入 ids 须为连续 CPU int64 [Q]，name 仅用于异常上下文；不限制重复值或数值范围。
// 无返回值，不修改张量；非法输入抛异常，供行号和槽号入口共同复用。
void check_ids(const at::Tensor& ids, const char* name) {
  TORCH_CHECK(ids.defined(), name, " must be defined");
  TORCH_CHECK(ids.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(ids.scalar_type() == at::kLong, name, " must be int64");
  TORCH_CHECK(ids.dim() == 1 && ids.is_contiguous(), name, " must be contiguous [Q]");
}

// 输入非负 layer_id 和连续 CPU int64 [Q] 行号，可重复；无返回值，非法形状／负数抛异常。
// 不修改张量。行号上限由 reader 按表行数检查，池本身不知道表大小。
void check_rows(std::int64_t layer_id, const at::Tensor& row_ids) {
  check_ids(row_ids, "row_ids");
  if (layer_id < 0) {
    throw std::out_of_range("Engram prefetch layer_id must be non-negative");
  }
  const auto* values = row_ids.data_ptr<std::int64_t>();
  for (std::int64_t i = 0; i < row_ids.numel(); ++i) {
    if (values[i] < 0) {
      throw std::out_of_range("Engram prefetch row_id must be non-negative");
    }
  }
}

// 输入非负 layer_id 和连续 CPU int64 [Q] 非负行号，可重复。
// 返回按首次出现顺序去重的行号副本；不修改输入，使 acquire 的每个槽在句柄内只占一个引用。
std::vector<std::int64_t> unique_rows(std::int64_t layer_id, const at::Tensor& row_ids) {
  check_rows(layer_id, row_ids);
  std::vector<std::int64_t> rows;
  rows.reserve(static_cast<std::size_t>(row_ids.numel()));
  std::unordered_set<std::int64_t> seen;
  seen.reserve(static_cast<std::size_t>(row_ids.numel()));
  const auto* values = row_ids.data_ptr<std::int64_t>();
  for (std::int64_t i = 0; i < row_ids.numel(); ++i) {
    if (seen.insert(values[i]).second) {
      rows.push_back(values[i]);
    }
  }
  return rows;
}

// 输入为只读整数向量；返回独立持有数据的连续 CPU int64 [Q]，不借用 vector 的地址。
// 空输入保留一维空张量；分配失败抛异常，不修改输入。
at::Tensor cpu_ids(const std::vector<std::int64_t>& values) {
  auto result = at::empty(
      {static_cast<std::int64_t>(values.size())}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  if (!values.empty()) {
    std::memcpy(result.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
  }
  return result;
}

// 分配一张 [rows,width] 的 pinned uint8 表：多申请 kAlignment-1 字节，再截取 4096 对齐的视图。
// storage 返回实际分配（持有内存），device_ptr 返回对齐视图起点在 GPU 侧的映射地址。
// 映射地址按 storage 起点查询后加同样的偏移，保证 CPU 视图与 GPU 地址指向同一字节。
// 4095B 是每张表显式申请的额外空间；pinned allocator 的取整／缓存开销另计，需在目标机测量。
at::Tensor aligned_table(
    std::int64_t rows,
    std::int64_t width,
    at::Tensor& storage,
    std::uintptr_t& device_ptr) {
  const auto bytes = rows * width;
  TORCH_CHECK(
      bytes <= std::numeric_limits<std::int64_t>::max() - (kAlignment - 1),
      "Engram prefetch aligned allocation exceeds int64 range");
  storage = at::empty(
      {bytes + kAlignment - 1},
      at::TensorOptions().device(at::kCPU).dtype(at::kByte).pinned_memory(true));
  const auto address = reinterpret_cast<std::uintptr_t>(storage.data_ptr<std::uint8_t>());
  const auto offset = (kAlignment - address % kAlignment) % kAlignment;
  void* mapped = nullptr;
  const auto result = cudaHostGetDevicePointer(&mapped, storage.data_ptr<std::uint8_t>(), 0);
  if (result != cudaSuccess) {
    throw std::runtime_error(
        std::string("Engram prefetch cudaHostGetDevicePointer: ") + cudaGetErrorString(result));
  }
  device_ptr = reinterpret_cast<std::uintptr_t>(mapped) + offset;
  return storage.narrow(0, static_cast<std::int64_t>(offset), bytes).view({rows, width});
}

}  // namespace

PrefetchRowHandle::PrefetchRowHandle(std::shared_ptr<PrefetchRowPool> owner)
    : owner_(std::move(owner)) {}

at::Tensor PrefetchRowHandle::missing_row_ids() const {
  return missing_row_ids_;
}

at::Tensor PrefetchRowHandle::missing_slot_ids() const {
  return missing_slot_ids_;
}

at::Tensor PrefetchRowHandle::row_ids() const {
  return row_ids_;
}

at::Tensor PrefetchRowHandle::slot_ids() const {
  return slot_ids_;
}

PrefetchRowPool::PrefetchRowPool(std::int64_t dim, std::int64_t pool_bytes, bool track_stats)
    : capacity_(pool_capacity(dim, pool_bytes)),
      weight_stride_(dim),
      scale_stride_(dim / 32),
      track_stats_(track_stats) {
  weight_ = aligned_table(capacity_, weight_stride_, weight_storage_, weight_device_ptr_);
  scale_ = aligned_table(capacity_, scale_stride_, scale_storage_, scale_device_ptr_);
  // 初始时所有槽都是 FREE，按槽号顺序串成空闲链表。
  slots_.resize(static_cast<std::size_t>(capacity_));
  for (std::int64_t i = 0; i < capacity_; ++i) {
    slots_[i].next = i + 1 < capacity_ ? i + 1 : -1;
  }
  free_head_ = 0;
  free_slots_ = capacity_;
}

at::Tensor PrefetchRowPool::weight() const {
  std::lock_guard lock(mutex_);
  require_open();
  return weight_;
}

at::Tensor PrefetchRowPool::scale() const {
  std::lock_guard lock(mutex_);
  require_open();
  return scale_;
}

std::uintptr_t PrefetchRowPool::weight_device_ptr() const {
  std::lock_guard lock(mutex_);
  require_open();
  return weight_device_ptr_;
}

std::uintptr_t PrefetchRowPool::scale_device_ptr() const {
  std::lock_guard lock(mutex_);
  require_open();
  return scale_device_ptr_;
}

std::int64_t PrefetchRowPool::capacity() const {
  return capacity_;
}

std::int64_t PrefetchRowPool::weight_stride() const {
  return weight_stride_;
}

std::int64_t PrefetchRowPool::scale_stride() const {
  return scale_stride_;
}

std::size_t PrefetchRowPool::RowKeyHash::operator()(const RowKey& key) const {
  const auto first = std::hash<std::int64_t>{}(key.layer_id);
  const auto second = std::hash<std::int64_t>{}(key.row_id);
  return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6) + (first >> 2));
}

void PrefetchRowPool::require_open() const {
  if (closed_) {
    throw std::runtime_error("Engram prefetch row pool is closed");
  }
}

void PrefetchRowPool::require_handle(const std::shared_ptr<PrefetchRowHandle>& handle) const {
  if (!handle || handle->owner_.get() != this) {
    throw std::invalid_argument("Engram prefetch handle belongs to another pool");
  }
}

void PrefetchRowPool::require_healthy(const Slot& slot) const {
  if (slot.failed) {
    throw std::runtime_error(
        "Engram prefetch row was cancelled before publication: layer=" +
        std::to_string(slot.key.layer_id) + ", row=" + std::to_string(slot.key.row_id));
  }
}

void PrefetchRowPool::remove_lru(std::int64_t slot_id) {
  auto& slot = slots_[slot_id];
  if (slot.previous >= 0) {
    slots_[slot.previous].next = slot.next;
  } else {
    lru_head_ = slot.next;
  }
  if (slot.next >= 0) {
    slots_[slot.next].previous = slot.previous;
  } else {
    lru_tail_ = slot.previous;
  }
  slot.previous = -1;
  slot.next = -1;
  --cached_slots_;
}

void PrefetchRowPool::add_lru(std::int64_t slot_id) {
  auto& slot = slots_[slot_id];
  slot.previous = -1;
  slot.next = lru_head_;
  if (lru_head_ >= 0) {
    slots_[lru_head_].previous = slot_id;
  } else {
    lru_tail_ = slot_id;
  }
  lru_head_ = slot_id;
  ++cached_slots_;
}

void PrefetchRowPool::pin(std::int64_t slot_id) {
  auto& slot = slots_[slot_id];
  // acquire 也能复用 PENDING；只有发布后的命中才计为使用。
  if (track_stats_ && slot.state == State::READY) {
    slot.used = true;
  }
  if (slot.references == 0) {
    if (slot.state == State::READY) {
      remove_lru(slot_id);
    }
    ++referenced_slots_;
  }
  ++slot.references;
}

// 回收一个无引用的槽：READY 先从 LRU 摘除；PENDING 已取消且没有在途 I/O。
void PrefetchRowPool::recycle(std::int64_t slot_id) {
  auto& slot = slots_[slot_id];
  if (slot.state == State::READY) {
    remove_lru(slot_id);
    --ready_slots_;
  } else {
    --pending_slots_;
  }
  index_.erase(slot.key);
  slot = Slot{};
  slot.next = free_head_;
  free_head_ = slot_id;
  ++free_slots_;
}

void PrefetchRowPool::unpin(std::int64_t slot_id, bool discard_pending) {
  auto& slot = slots_[slot_id];
  --slot.references;
  if (slot.references == 0) {
    --referenced_slots_;
    // READY 进入 LRU；PENDING 只有确认无在途写入时才可归还空闲链表。
    if (slot.state == State::READY) {
      add_lru(slot_id);
    } else if (slot.failed || discard_pending) {
      recycle(slot_id);
    }
  }
}

std::int64_t PrefetchRowPool::take_slot() {
  // 调用方已检查过 free_slots_ + 可回收的 LRU 槽足够，所以这里 lru_tail_ 一定有效。
  if (free_head_ < 0) {
    if (track_stats_) {
      // LRU 只含零引用 READY 槽，在回收清空元数据前统计此次淘汰。
      ++counters_.evicted_slots;
      counters_.evicted_unused_slots += !slots_[lru_tail_].used;
    }
    recycle(lru_tail_);
  }
  const auto slot_id = free_head_;
  free_head_ = slots_[slot_id].next;
  slots_[slot_id].next = -1;
  --free_slots_;
  return slot_id;
}

void PrefetchRowPool::release_slots(PrefetchRowHandle& handle, bool discard_pending) {
  for (const auto slot_id : handle.slots_) {
    unpin(slot_id, discard_pending);
  }
  handle.released_ = true;
}

std::shared_ptr<PrefetchRowHandle> PrefetchRowPool::acquire(
    std::int64_t layer_id, const at::Tensor& row_ids) {
  // 去重和行号校验不需要池状态，放在锁外做。
  const auto rows = unique_rows(layer_id, row_ids);
  std::lock_guard lock(mutex_);
  require_open();
  // 第一步：把输入行分成"池中已有"和"需要新建"两类，此时不修改任何状态。
  std::vector<std::int64_t> existing(rows.size(), -1);
  std::vector<std::int64_t> missing;
  missing.reserve(rows.size());
  std::int64_t protected_cached = 0;
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const auto found = index_.find(RowKey{layer_id, rows[i]});
    if (found == index_.end()) {
      missing.push_back(rows[i]);
    } else {
      const auto slot_id = found->second;
      require_healthy(slots_[slot_id]);
      existing[i] = slot_id;
      if (slots_[slot_id].state == State::READY && slots_[slot_id].references == 0) {
        ++protected_cached;
      }
    }
  }
  // 第二步：检查空间。可用槽 = 空闲槽 + LRU 中的缓存槽，但本次要复用的缓存槽稍后会被 pin，
  // 不能算作可回收空间。
  const auto available = free_slots_ + cached_slots_ - protected_cached;
  const auto absent = static_cast<std::int64_t>(missing.size());
  if (absent > available) {
    return nullptr;
  }
  auto handle = std::shared_ptr<PrefetchRowHandle>(new PrefetchRowHandle(shared_from_this()));
  handle->slots_.reserve(rows.size());
  handle->created_slots_.reserve(missing.size());
  handle->missing_row_ids_ = cpu_ids(missing);
  handle->missing_slot_ids_ = at::empty(
      {static_cast<std::int64_t>(missing.size())}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  handle->row_ids_ = cpu_ids(rows);
  handle->slot_ids_ = cpu_ids(existing);
  auto* slot_ids = handle->slot_ids_.data_ptr<std::int64_t>();
  auto* missing_slots = handle->missing_slot_ids_.data_ptr<std::int64_t>();
  try {
    // 第三步：先 pin 全部已有行，使它们离开 LRU；再为新行取槽，
    // 这样 take_slot 回收 LRU 尾部时不会淘汰本次要复用的行。
    for (const auto slot_id : existing) {
      if (slot_id >= 0) {
        pin(slot_id);
        handle->slots_.push_back(slot_id);
      }
    }
    std::size_t next = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (existing[i] >= 0) {
        continue;
      }
      const auto slot_id = take_slot();
      auto& slot = slots_[slot_id];
      slot.key = RowKey{layer_id, rows[i]};
      slot.state = State::PENDING;
      slot.references = 1;
      ++referenced_slots_;
      ++pending_slots_;
      handle->slots_.push_back(slot_id);
      handle->created_slots_.push_back(slot_id);
      slot_ids[i] = slot_id;
      missing_slots[next++] = slot_id;
      index_.emplace(slot.key, slot_id);
    }
  } catch (...) {
    // 中途失败（例如内存分配异常）：撤销已新建的槽并归还已加的引用，再抛出原异常。
    for (const auto slot_id : handle->created_slots_) {
      slots_[slot_id].failed = true;
    }
    release_slots(*handle);
    throw;
  }
  if (track_stats_) {
    // 统计"本次无需新读"的行数（批内重复 + 池中已有）；实际提交读取的行数由调用方 record_submitted。
    counters_.dedup_saved_rows += row_ids.numel() - absent;
  }
  return handle;
}

std::int64_t PrefetchRowPool::available_slots() const {
  std::lock_guard lock(mutex_);
  require_open();
  return free_slots_ + cached_slots_;
}

std::int64_t PrefetchRowPool::slots_needed(
    std::int64_t layer_id, const at::Tensor& row_ids, std::int64_t limit) const {
  check_rows(layer_id, row_ids);
  const auto* rows = row_ids.data_ptr<std::int64_t>();
  std::unordered_set<std::int64_t> seen;
  std::lock_guard lock(mutex_);
  require_open();
  std::int64_t needed = 0;
  for (std::int64_t i = 0; i < row_ids.numel(); ++i) {
    const auto found = index_.find(RowKey{layer_id, rows[i]});
    // 本段也需要的 LRU 槽必须保留，与尚无槽的行一样占用可用空间。
    const bool needs_slot = found == index_.end() ||
        (slots_[found->second].state == State::READY && slots_[found->second].references == 0);
    if (needs_slot && seen.insert(rows[i]).second) {
      ++needed;
      if (limit >= 0 && needed > limit) {
        break;
      }
    }
  }
  return needed;
}

at::Tensor PrefetchRowPool::pending_slots(std::int64_t layer_id, const at::Tensor& row_ids) const {
  check_rows(layer_id, row_ids);
  auto result = at::full({row_ids.numel()}, -1, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  const auto* rows = row_ids.data_ptr<std::int64_t>();
  auto* result_ids = result.data_ptr<std::int64_t>();
  std::lock_guard lock(mutex_);
  require_open();
  for (std::int64_t i = 0; i < row_ids.numel(); ++i) {
    const auto found = index_.find(RowKey{layer_id, rows[i]});
    if (found != index_.end() && slots_[found->second].state == State::PENDING) {
      require_healthy(slots_[found->second]);
      result_ids[i] = found->second;
    }
  }
  return result;
}

bool PrefetchRowPool::is_ready(const std::shared_ptr<PrefetchRowHandle>& handle) const {
  std::lock_guard lock(mutex_);
  require_open();
  require_handle(handle);
  if (handle->released_) {
    throw std::runtime_error("Engram prefetch handle is released");
  }
  bool ready = true;
  for (const auto slot_id : handle->slots_) {
    require_healthy(slots_[slot_id]);
    ready = ready && slots_[slot_id].state == State::READY;
  }
  return ready;
}

void PrefetchRowPool::mark_ready(const at::Tensor& slot_ids) {
  check_ids(slot_ids, "slot_ids");
  std::lock_guard lock(mutex_);
  require_open();
  // 第一遍只检查、收集去重后的槽号；任一槽不合法就抛异常，不修改任何状态。
  std::unordered_set<std::int64_t> seen;
  seen.reserve(static_cast<std::size_t>(slot_ids.numel()));
  std::vector<std::int64_t> unique;
  unique.reserve(static_cast<std::size_t>(slot_ids.numel()));
  const auto* ids = slot_ids.data_ptr<std::int64_t>();
  for (std::int64_t i = 0; i < slot_ids.numel(); ++i) {
    const auto slot_id = ids[i];
    if (slot_id < 0 || slot_id >= capacity_) {
      throw std::out_of_range("Engram prefetch slot is outside pool capacity");
    }
    require_healthy(slots_[slot_id]);
    if (slots_[slot_id].state != State::PENDING) {
      throw std::runtime_error("Engram prefetch mark_ready requires PENDING slots");
    }
    if (seen.insert(slot_id).second) {
      unique.push_back(slot_id);
    }
  }
  for (const auto slot_id : unique) {
    // 第二遍发布：此时数据已写完。无引用的槽直接进入 LRU，可被后续请求复用或回收。
    auto& slot = slots_[slot_id];
    slot.state = State::READY;
    if (track_stats_) {
      slot.used = false;
    }
    --pending_slots_;
    ++ready_slots_;
    if (slot.references == 0) {
      add_lru(slot_id);
    }
  }
}

PrefetchRowResolution PrefetchRowPool::resolve(std::int64_t layer_id, const at::Tensor& row_ids) {
  // 先校验输入并在锁外分配输出张量，缩短持锁时间；非法输入不会产生任何引用。
  check_rows(layer_id, row_ids);
  PrefetchRowResolution result;
  result.slot_ids = at::full({row_ids.numel()}, -1, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  result.hit_mask = at::zeros({row_ids.numel()}, at::TensorOptions().device(at::kCPU).dtype(at::kBool));
  result.handle = std::shared_ptr<PrefetchRowHandle>(new PrefetchRowHandle(shared_from_this()));
  result.handle->slots_.reserve(static_cast<std::size_t>(row_ids.numel()));
  result.handle->missing_row_ids_ = cpu_ids({});
  result.handle->missing_slot_ids_ = cpu_ids({});
  result.handle->row_ids_ = result.handle->missing_row_ids_;
  result.handle->slot_ids_ = result.handle->missing_slot_ids_;
  std::unordered_set<std::int64_t> pinned;
  pinned.reserve(static_cast<std::size_t>(row_ids.numel()));
  std::lock_guard lock(mutex_);
  require_open();
  const auto* rows = row_ids.data_ptr<std::int64_t>();
  auto* slot_ids = result.slot_ids.data_ptr<std::int64_t>();
  auto* mask = result.hit_mask.data_ptr<bool>();
  try {
    for (std::int64_t i = 0; i < row_ids.numel(); ++i) {
      const auto found = index_.find(RowKey{layer_id, rows[i]});
      if (found == index_.end()) {
        continue;
      }
      const auto slot_id = found->second;
      if (slots_[slot_id].state != State::READY) {
        continue;  // 只有 READY 可复制到 batch 暂存区。
      }
      // 同一槽可能被多个查询位置命中，只 pin 一次，避免句柄里出现重复引用。
      if (pinned.insert(slot_id).second) {
        pin(slot_id);
        result.handle->slots_.push_back(slot_id);
      }
      slot_ids[i] = slot_id;
      mask[i] = true;
    }
  } catch (...) {
    release_slots(*result.handle);
    throw;
  }
  return result;
}

void PrefetchRowPool::release(const std::shared_ptr<PrefetchRowHandle>& handle, bool discard_pending) {
  std::lock_guard lock(mutex_);
  require_handle(handle);
  if (handle->released_) {
    return;
  }
  require_open();
  release_slots(*handle, discard_pending);
}

void PrefetchRowPool::cancel(const std::shared_ptr<PrefetchRowHandle>& handle) {
  std::lock_guard lock(mutex_);
  require_handle(handle);
  if (handle->released_) {
    return;
  }
  require_open();
  // 只把本句柄新建、仍未读完的槽标记为失败；unpin 时无其他引用的失败槽会立即回收。
  for (const auto slot_id : handle->created_slots_) {
    if (slots_[slot_id].state == State::PENDING) {
      slots_[slot_id].failed = true;
    }
  }
  release_slots(*handle);
}

PrefetchRowPoolStats PrefetchRowPool::stats() const {
  std::lock_guard lock(mutex_);
  auto result = counters_;
  result.capacity = capacity_;
  result.payload_bytes = capacity_ * (weight_stride_ + scale_stride_);
  result.occupied_slots = capacity_ - free_slots_;
  result.referenced_slots = referenced_slots_;
  result.pending_slots = pending_slots_;
  result.ready_slots = ready_slots_;
  result.cached_slots = cached_slots_;
  result.tracking_enabled = track_stats_;
  result.closed = closed_;
  return result;
}

void PrefetchRowPool::record_submitted(std::int64_t rows) {
  if (!track_stats_) {
    return;
  }
  if (rows < 0) {
    throw std::invalid_argument("Engram prefetch submitted rows must be non-negative");
  }
  std::lock_guard lock(mutex_);
  require_open();
  counters_.submitted_rows += rows;
}

void PrefetchRowPool::record_forward(std::int64_t hits, std::int64_t misses, double miss_ms) {
  if (!track_stats_) {
    return;
  }
  if (hits < 0 || misses < 0 || miss_ms < 0 || !std::isfinite(miss_ms)) {
    throw std::invalid_argument("Engram prefetch forward counters must be finite and non-negative");
  }
  std::lock_guard lock(mutex_);
  require_open();
  counters_.forward_hits += hits;
  counters_.forward_misses += misses;
  counters_.miss_read_ms += miss_ms;
}

void PrefetchRowPool::record_deferred(std::int64_t count) {
  if (!track_stats_) {
    return;
  }
  if (count < 0) {
    throw std::invalid_argument("Engram prefetch deferred count must be non-negative");
  }
  std::lock_guard lock(mutex_);
  require_open();
  counters_.deferred_requests += count;
}

void PrefetchRowPool::close() {
  // 在锁内把 Tensor 移到局部变量，函数返回时才析构，pinned 内存的释放不占用池锁。
  at::Tensor weight, scale, weight_storage, scale_storage;
  {
    std::lock_guard lock(mutex_);
    if (closed_) {
      return;
    }
    if (referenced_slots_ != 0 || pending_slots_ != 0) {
      throw std::runtime_error("Engram prefetch close requires released handles and completed reads");
    }
    closed_ = true;
    weight = std::move(weight_);
    scale = std::move(scale_);
    weight_storage = std::move(weight_storage_);
    scale_storage = std::move(scale_storage_);
    weight_device_ptr_ = 0;
    scale_device_ptr_ = 0;
    decltype(index_){}.swap(index_);
    decltype(slots_){}.swap(slots_);
    free_head_ = lru_head_ = lru_tail_ = -1;
    free_slots_ = capacity_;
    ready_slots_ = cached_slots_ = 0;
  }
}

}  // namespace sglang_engram_ssd::query
