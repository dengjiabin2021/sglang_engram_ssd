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

std::int64_t pool_capacity(std::int64_t dim, std::int64_t pool_bytes) {
  if (dim <= 0 || dim % 32 != 0 || dim > std::numeric_limits<std::int64_t>::max() - dim / 32) {
    throw std::invalid_argument("Engram prefetch dim must be a positive multiple of 32");
  }
  if (pool_bytes <= 0 || pool_bytes / (dim + dim / 32) == 0) {
    throw std::invalid_argument("Engram prefetch pool_bytes must hold at least one row");
  }
  return pool_bytes / (dim + dim / 32);
}

void check_ids(const at::Tensor& ids, const char* name) {
  TORCH_CHECK(ids.defined(), name, " must be defined");
  TORCH_CHECK(ids.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(ids.scalar_type() == at::kLong, name, " must be int64");
  TORCH_CHECK(ids.dim() == 1 && ids.is_contiguous(), name, " must be contiguous [Q]");
}

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

at::Tensor cpu_ids(const std::vector<std::int64_t>& values) {
  auto result = at::empty(
      {static_cast<std::int64_t>(values.size())}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  if (!values.empty()) {
    std::memcpy(result.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
  }
  return result;
}

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

PrefetchRowPool::PrefetchRowPool(std::int64_t dim, std::int64_t pool_bytes, bool track_stats)
    : capacity_(pool_capacity(dim, pool_bytes)),
      weight_stride_(dim),
      scale_stride_(dim / 32),
      track_stats_(track_stats) {
  weight_ = aligned_table(capacity_, weight_stride_, weight_storage_, weight_device_ptr_);
  scale_ = aligned_table(capacity_, scale_stride_, scale_storage_, scale_device_ptr_);
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
  if (slot.references == 0) {
    if (slot.state == State::READY) {
      remove_lru(slot_id);
    }
    ++referenced_slots_;
  }
  ++slot.references;
}

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

void PrefetchRowPool::unpin(std::int64_t slot_id) {
  auto& slot = slots_[slot_id];
  --slot.references;
  if (slot.references == 0) {
    --referenced_slots_;
    if (slot.state == State::READY) {
      add_lru(slot_id);
    } else if (slot.failed) {
      recycle(slot_id);
    }
  }
}

std::int64_t PrefetchRowPool::take_slot() {
  if (free_head_ < 0) {
    recycle(lru_tail_);
  }
  const auto slot_id = free_head_;
  free_head_ = slots_[slot_id].next;
  slots_[slot_id].next = -1;
  --free_slots_;
  return slot_id;
}

void PrefetchRowPool::release_slots(PrefetchRowHandle& handle) {
  for (const auto slot_id : handle.slots_) {
    unpin(slot_id);
  }
  handle.released_ = true;
}

std::shared_ptr<PrefetchRowHandle> PrefetchRowPool::acquire(
    std::int64_t layer_id, const at::Tensor& row_ids) {
  const auto rows = unique_rows(layer_id, row_ids);
  std::lock_guard lock(mutex_);
  require_open();
  std::vector<std::int64_t> existing;
  std::vector<std::int64_t> missing;
  existing.reserve(rows.size());
  missing.reserve(rows.size());
  std::int64_t protected_cached = 0;
  for (const auto row : rows) {
    const auto found = index_.find(RowKey{layer_id, row});
    if (found == index_.end()) {
      missing.push_back(row);
    } else {
      const auto slot_id = found->second;
      require_healthy(slots_[slot_id]);
      existing.push_back(slot_id);
      if (slots_[slot_id].state == State::READY && slots_[slot_id].references == 0) {
        ++protected_cached;
      }
    }
  }
  // 本次也需要的缓存行先排除，不能为了新行将它们当作可淘汰空间。
  if (static_cast<std::int64_t>(missing.size()) > free_slots_ + cached_slots_ - protected_cached) {
    return nullptr;
  }
  auto handle = std::shared_ptr<PrefetchRowHandle>(new PrefetchRowHandle(shared_from_this()));
  handle->slots_.reserve(rows.size());
  handle->created_slots_.reserve(missing.size());
  handle->missing_row_ids_ = cpu_ids(missing);
  handle->missing_slot_ids_ = at::empty(
      {static_cast<std::int64_t>(missing.size())}, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  auto* missing_slots = handle->missing_slot_ids_.data_ptr<std::int64_t>();
  try {
    // 先固定全部现有行，再逐项回收 LRU，避免新行淘汰本次仍要用的缓存行。
    for (const auto slot_id : existing) {
      pin(slot_id);
      handle->slots_.push_back(slot_id);
    }
    for (std::size_t i = 0; i < missing.size(); ++i) {
      const auto slot_id = take_slot();
      auto& slot = slots_[slot_id];
      slot.key = RowKey{layer_id, missing[i]};
      slot.state = State::PENDING;
      slot.references = 1;
      ++referenced_slots_;
      ++pending_slots_;
      handle->slots_.push_back(slot_id);
      handle->created_slots_.push_back(slot_id);
      missing_slots[i] = slot_id;
      index_.emplace(slot.key, slot_id);
    }
  } catch (...) {
    for (const auto slot_id : handle->created_slots_) {
      slots_[slot_id].failed = true;
    }
    release_slots(*handle);
    throw;
  }
  if (track_stats_) {
    counters_.submitted_rows += static_cast<std::int64_t>(missing.size());
    counters_.dedup_saved_rows += row_ids.numel() - static_cast<std::int64_t>(missing.size());
  }
  return handle;
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
    auto& slot = slots_[slot_id];
    slot.state = State::READY;
    --pending_slots_;
    ++ready_slots_;
    if (slot.references == 0) {
      add_lru(slot_id);
    }
  }
}

PrefetchRowResolution PrefetchRowPool::resolve(std::int64_t layer_id, const at::Tensor& row_ids) {
  // 同样先校验全部行，非法输入不产生部分引用。
  check_rows(layer_id, row_ids);
  PrefetchRowResolution result;
  result.slot_ids = at::full({row_ids.numel()}, -1, at::TensorOptions().device(at::kCPU).dtype(at::kLong));
  result.hit_mask = at::zeros({row_ids.numel()}, at::TensorOptions().device(at::kCPU).dtype(at::kBool));
  result.handle = std::shared_ptr<PrefetchRowHandle>(new PrefetchRowHandle(shared_from_this()));
  result.handle->slots_.reserve(static_cast<std::size_t>(row_ids.numel()));
  result.handle->missing_row_ids_ = cpu_ids({});
  result.handle->missing_slot_ids_ = cpu_ids({});
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
      require_healthy(slots_[slot_id]);
      if (slots_[slot_id].state != State::READY) {
        continue;
      }
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

void PrefetchRowPool::release(const std::shared_ptr<PrefetchRowHandle>& handle) {
  std::lock_guard lock(mutex_);
  require_handle(handle);
  if (handle->released_) {
    return;
  }
  require_open();
  release_slots(*handle);
}

void PrefetchRowPool::cancel(const std::shared_ptr<PrefetchRowHandle>& handle) {
  std::lock_guard lock(mutex_);
  require_handle(handle);
  if (handle->released_) {
    return;
  }
  require_open();
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
  // pinned allocator 的回收在锁外完成，不与 acquire/resolve 的状态锁交织。
}

}  // namespace sglang_engram_ssd::query
