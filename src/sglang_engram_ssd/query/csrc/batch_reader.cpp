#include "batch_reader.h"

#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <exception>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <sys/types.h>
#include <unistd.h>

namespace sglang_engram_ssd::query {

BatchRowReader::BatchRowReader(
    const std::vector<RowTableSpec>& tables,
    std::int64_t cache_bytes,
    std::optional<std::int64_t> num_threads,
    std::optional<std::int64_t> thread_pool_queue_depth,
    std::optional<std::int64_t> io_uring_queue_depth,
    const std::string& io_backend)
    : cache_budget_bytes_(cache_bytes) {
  if (cache_bytes < 0) {
    throw std::invalid_argument("Engram SSD cache_bytes must be non-negative");
  }
  if (io_backend != "pread" && io_backend != "io_uring") {
    throw std::invalid_argument("io_backend must be 'pread' or 'io_uring'");
  }
  // 初始化时按层号保存元数据；这里只借用 fd，不打开或复制权重文件。
  tables_.reserve(tables.size());
  for (const auto& table : tables) {
    if (!tables_.emplace(table.layer_id, table).second) {
      throw std::invalid_argument(
          "Duplicate Engram SSD layer: " + std::to_string(table.layer_id));
    }
  }
  if (io_backend == "io_uring") {
    // 线程池参数不会参与 ring 的构造；即使其值不合法，也只提示并忽略。
    if (num_threads.has_value()) {
      TORCH_WARN("Engram SSD io_uring ignores num_threads");
    }
    if (thread_pool_queue_depth.has_value()) {
      TORCH_WARN("Engram SSD io_uring ignores thread_pool_queue_depth");
    }
    const auto depth = io_uring_queue_depth.value_or(32);
    if (depth <= 0) {
      throw std::invalid_argument("io_uring_queue_depth must be positive");
    }
    io_uring_reader_ = std::make_unique<IoUringReader>(depth);
  } else {
    if (io_uring_queue_depth.has_value()) {
      TORCH_WARN("Engram SSD pread ignores io_uring_queue_depth");
    }
    const auto threads = num_threads.value_or(0);
    if (threads < 0) {
      throw std::invalid_argument("pread num_threads must be non-negative");
    }
    if (threads > 0) {
      const auto depth = thread_pool_queue_depth.value_or(32);
      if (depth <= 0) {
        throw std::invalid_argument("thread_pool_queue_depth must be positive");
      }
      thread_pool_ = std::make_unique<ThreadPool>(threads, depth);
    } else if (thread_pool_queue_depth.has_value()) {
      // 保留底层串行 pread 接口；串行读取不使用等待任务队列。
      TORCH_WARN("Engram SSD serial pread ignores thread_pool_queue_depth");
    }
  }
}

BatchRowReader::~BatchRowReader() {
  close();
}

std::pair<at::Tensor, at::Tensor> BatchRowReader::allocate_workspace(
    std::int64_t layer_id,
    std::int64_t capacity) const {
  const auto& table = open_table(layer_id);
  TORCH_CHECK(capacity > 0, "Workspace capacity must be positive");

  // 分配的是 CPU 页锁定内存，不是 GPU 内存；失败时直接抛错，不回退到普通内存。
  const auto options = at::TensorOptions()
                           .device(at::kCPU)
                           .dtype(c10::ScalarType::Byte)
                           .pinned_memory(true);
  auto weight = at::empty({capacity, table.dim}, options);
  auto scale = at::empty({capacity, table.dim / 32}, options);

  // 返回 Tensor 句柄，不复制底层数据。工作区由调用方持有，不占用行缓存预算。
  return {std::move(weight), std::move(scale)};
}

void BatchRowReader::read_rows(
    std::int64_t layer_id,
    const at::Tensor& row_ids,
    at::Tensor& weight_out,
    at::Tensor& scale_out) {
  // 1. 选择当前层，并在访问原始内存前检查输入和输出。
  const RowTableSpec& table = open_table(layer_id);
  check_inputs(table, row_ids, weight_out, scale_out);

  const auto query_count = row_ids.size(0);
  if (query_count == 0) {
    return;  // 空批次不获取数据指针，也不发起 I/O。
  }

  // 2. 取得调用方工作区的真实地址；pinned Tensor 的指针可直接用于 pread/memcpy。
  // 不创建 pageable 临时输出，不再经 Python bytearray/frombuffer 逐行中转。
  // slot 表示本批输出位置，row_id 表示原表全局行号。
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;
  const auto* ids = row_ids.data_ptr<std::int64_t>();
  auto* weights = weight_out.data_ptr<std::uint8_t>();
  auto* scales = scale_out.data_ptr<std::uint8_t>();

  if (io_uring_reader_) {
    read_rows_uring(table, ids, query_count, weights, scales);
    return;
  }
  if (thread_pool_) {
    read_rows_threaded(table, ids, query_count, weights, scales);
    return;
  }

  // 3. 每次查询占一个输出槽；按输入顺序处理，不排序、不去重。
  for (std::int64_t slot = 0; slot < query_count; ++slot) {
    const auto row_id = ids[slot];
    if (row_id < 0 || row_id >= table.rows) {
      throw std::out_of_range(
          "Engram SSD row " + std::to_string(row_id) +
          " is outside layer " + std::to_string(layer_id));
    }

    auto* weight_row = weights + slot * weight_bytes;
    auto* scale_row = scales + slot * scale_bytes;

    // 先查 C++ 缓存；重复查询仍分别填充各自的输出槽，不合并查询位置。
    const auto* cached = find_cached_row(layer_id, row_id);
    if (cached != nullptr) {
      std::memcpy(weight_row, cached->data(), static_cast<std::size_t>(weight_bytes));
      std::memcpy(
          scale_row,
          cached->data() + weight_bytes,
          static_cast<std::size_t>(scale_bytes));
      continue;
    }

    read_file_row(table, row_id, weight_row, scale_row);
    // 两段数据都读满才入缓存，读取失败不会留下半行数据。
    cache_row(layer_id, row_id, weight_row, scale_row);
  }
}

void BatchRowReader::close() {
  // 先回收 I/O 后端，再清理缓存和表信息，不关闭借用的模型 fd。
  // 线程池排空任务后退出；io_uring 的在途请求已在 read_rows 返回前收齐。
  thread_pool_.reset();
  io_uring_reader_.reset();
  cache_index_.clear();
  cache_lru_.clear();
  cached_bytes_ = 0;
  tables_.clear();
  closed_ = true;
}

const RowTableSpec& BatchRowReader::open_table(std::int64_t layer_id) const {
  if (closed_) {
    throw std::runtime_error("Engram SSD reader is closed");
  }
  const auto table = tables_.find(layer_id);
  if (table == tables_.end()) {
    throw std::out_of_range(
        "Engram SSD table is not initialized for layer " +
        std::to_string(layer_id));
  }
  return table->second;
}

void BatchRowReader::check_inputs(
    const RowTableSpec& table,
    const at::Tensor& row_ids,
    const at::Tensor& weight_out,
    const at::Tensor& scale_out) const {
  // 三个张量必须已定义、位于 CPU，且内存连续，才能直接访问其数据指针。
  TORCH_CHECK(
      row_ids.defined() && weight_out.defined() && scale_out.defined(),
      "Input and output tensors must be defined");
  TORCH_CHECK(
      row_ids.device().is_cpu() && weight_out.device().is_cpu() &&
          scale_out.device().is_cpu(),
      "Input and output tensors must be on CPU");
  TORCH_CHECK(
      row_ids.is_contiguous() && weight_out.is_contiguous() && scale_out.is_contiguous(),
      "Input and output tensors must be contiguous");

  // 行号使用 int64；输出保存原始字节，因此使用 uint8，不在此处反量化。
  TORCH_CHECK(row_ids.scalar_type() == c10::ScalarType::Long, "row_ids must be int64");
  TORCH_CHECK(
      weight_out.scalar_type() == c10::ScalarType::Byte &&
          scale_out.scalar_type() == c10::ScalarType::Byte,
      "Output tensors must be uint8");

  // 对应接口约定：[Q]、[K,D]、[K,D/32]，并且 Q <= K。
  TORCH_CHECK(row_ids.dim() == 1, "row_ids must have shape [Q]");
  TORCH_CHECK(
      weight_out.dim() == 2 && scale_out.dim() == 2,
      "Output tensors must be two-dimensional");
  TORCH_CHECK(
      weight_out.size(0) == scale_out.size(0), "Output capacities must match");
  TORCH_CHECK(weight_out.size(1) == table.dim, "weight_out width must match table dim");
  TORCH_CHECK(
      scale_out.size(1) == table.dim / 32, "scale_out width must match table dim / 32");
  TORCH_CHECK(row_ids.size(0) <= weight_out.size(0), "Query count exceeds output capacity");
}

std::int64_t BatchRowReader::row_offset(
    std::int64_t tensor_offset,
    std::int64_t row_id,
    std::int64_t row_bytes) const {
  const auto max_offset =
      static_cast<std::int64_t>(std::numeric_limits<off_t>::max());
  if (tensor_offset < 0 || tensor_offset > max_offset || row_id < 0 || row_bytes <= 0) {
    throw std::invalid_argument("Invalid Engram SSD row address");
  }

  // 先检查乘加是否安全，再计算行起点。
  const auto available = max_offset - tensor_offset;
  if (row_id > available / row_bytes) {
    throw std::overflow_error("Engram SSD row address exceeds off_t range");
  }
  const auto offset = tensor_offset + row_id * row_bytes;

  // 行尾也必须落在文件偏移类型可表示的范围内。
  if (row_bytes - 1 > max_offset - offset) {
    throw std::overflow_error("Engram SSD row end exceeds off_t range");
  }
  return offset;
}

void BatchRowReader::read_exact_at(
    int fd,
    std::uint8_t* destination,
    std::int64_t bytes,
    std::int64_t offset) const {
  std::int64_t received = 0;
  while (received < bytes) {
    // 每轮从尚未读到的位置继续，目标内存和文件偏移同步向后移动。
    const auto result = ::pread(
        fd,
        destination + received,
        static_cast<std::size_t>(bytes - received),
        static_cast<off_t>(offset + received));

    if (result > 0) {
      received += result;
      continue;
    }
    if (result == 0) {
      throw std::runtime_error(
          "Engram SSD unexpected EOF: fd=" + std::to_string(fd) +
          ", offset=" + std::to_string(offset) +
          ", expected=" + std::to_string(bytes) +
          ", received=" + std::to_string(received));
    }

    // result < 0 表示系统调用失败。信号中断可重试，其余错误保留 errno 抛出。
    const int error = errno;
    if (error == EINTR) {
      continue;
    }
    throw std::system_error(
        error,
        std::generic_category(),
        "Engram SSD pread: fd=" + std::to_string(fd) +
            ", offset=" + std::to_string(offset + received));
  }
}

const std::vector<std::uint8_t>* BatchRowReader::find_cached_row(
    std::int64_t layer_id,
    std::int64_t row_id,
    bool update_lru) {
  const auto layer = cache_index_.find(layer_id);
  if (layer == cache_index_.end()) {
    return nullptr;
  }
  const auto row = layer->second.find(row_id);
  if (row == layer->second.end()) {
    return nullptr;
  }

  // splice 只移动链表节点，不复制行数据，原有迭代器保持有效。
  const auto cached = row->second;
  if (update_lru) {
    cache_lru_.splice(cache_lru_.begin(), cache_lru_, cached);
  }
  return &cached->data;
}

void BatchRowReader::cache_row(
    std::int64_t layer_id,
    std::int64_t row_id,
    const std::uint8_t* weight,
    const std::uint8_t* scale) {
  const auto& table = open_table(layer_id);
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;

  // 不够容纳一整行时直接跳过，包括预算为 0 的情况；不创建临时缓存行。
  if (weight_bytes > cache_budget_bytes_ ||
      scale_bytes > cache_budget_bytes_ - weight_bytes) {
    return;
  }
  const auto row_bytes = weight_bytes + scale_bytes;

  // 预算由所有层共用。用减法比较剩余空间，避免 cached_bytes_ + row_bytes 溢出。
  while (cached_bytes_ > cache_budget_bytes_ - row_bytes) {
    evict_oldest_row();
  }

  std::vector<std::uint8_t> data(static_cast<std::size_t>(row_bytes));
  std::memcpy(data.data(), weight, static_cast<std::size_t>(weight_bytes));
  std::memcpy(
      data.data() + weight_bytes, scale, static_cast<std::size_t>(scale_bytes));

  auto& index = cache_index_[layer_id];
  cache_lru_.push_front(CachedRow{layer_id, row_id, std::move(data)});
  try {
    if (!index.emplace(row_id, cache_lru_.begin()).second) {
      cache_lru_.pop_front();
      return;
    }
  } catch (...) {
    // 索引插入失败时撤销新节点，避免链表与索引不一致。
    cache_lru_.pop_front();
    throw;
  }
  cached_bytes_ += row_bytes;
}

void BatchRowReader::evict_oldest_row() {
  const auto& oldest = cache_lru_.back();
  auto& index = cache_index_.at(oldest.layer_id);
  index.erase(oldest.row_id);
  cached_bytes_ -= static_cast<std::int64_t>(oldest.data.size());
  cache_lru_.pop_back();
}

void BatchRowReader::read_file_row(
    const RowTableSpec& table,
    std::int64_t row_id,
    std::uint8_t* weight,
    std::uint8_t* scale) const {
  // 两段数据分别寻址，允许位于不同文件；这里不访问 LRU 或 Python 对象。
  const auto weight_offset = row_offset(table.weight_offset, row_id, table.dim);
  const auto scale_offset = row_offset(table.scale_offset, row_id, table.dim / 32);
  read_exact_at(table.weight_fd, weight, table.dim, weight_offset);
  read_exact_at(table.scale_fd, scale, table.dim / 32, scale_offset);
}

void BatchRowReader::read_rows_threaded(
    const RowTableSpec& table,
    const std::int64_t* row_ids,
    std::int64_t query_count,
    std::uint8_t* weights,
    std::uint8_t* scales) {
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;
  std::vector<std::future<void>> pending;
  // 提交任务前预留跟踪空间，避免提交成功后因 vector 扩容失败而丢失 future。
  pending.reserve(static_cast<std::size_t>(query_count));
  std::exception_ptr failure;

  try {
    for (std::int64_t slot = 0; slot < query_count; ++slot) {
      const auto row_id = row_ids[slot];
      if (row_id < 0 || row_id >= table.rows) {
        throw std::out_of_range(
            "Engram SSD row " + std::to_string(row_id) +
            " is outside layer " + std::to_string(table.layer_id));
      }
      auto* weight_row = weights + slot * weight_bytes;
      auto* scale_row = scales + slot * scale_bytes;

      // 提交期间只查看当前缓存，不移动或淘汰节点，worker 也不访问缓存。
      const auto* cached = find_cached_row(table.layer_id, row_id, false);
      if (cached != nullptr) {
        std::memcpy(weight_row, cached->data(), static_cast<std::size_t>(weight_bytes));
        std::memcpy(
            scale_row, cached->data() + weight_bytes, static_cast<std::size_t>(scale_bytes));
        continue;
      }

      // 每项任务只写自己的槽位；复制表元数据，不保留 Python 对象或 Tensor 引用。
      pending.push_back(thread_pool_->submit(
          [this, table, row_id, weight_row, scale_row] {
            read_file_row(table, row_id, weight_row, scale_row);
          }));
    }
  } catch (...) {
    failure = std::current_exception();
  }

  // 包括提交失败、非法行号及读取失败：都等已提交任务结束后再向调用方抛错。
  for (auto& task : pending) {
    try {
      task.get();
    } catch (...) {
      if (!failure) {
        failure = std::current_exception();
      }
    }
  }
  if (failure) {
    std::rethrow_exception(failure);
  }

  commit_cache_rows(table, row_ids, query_count, weights, scales);
}

void BatchRowReader::read_rows_uring(
    const RowTableSpec& table,
    const std::int64_t* row_ids,
    std::int64_t query_count,
    std::uint8_t* weights,
    std::uint8_t* scales) {
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;
  std::vector<IoReadRequest> requests;
  requests.reserve(static_cast<std::size_t>(query_count) * 2);

  // 在提交任何 I/O 前完成参数检查和请求描述符构造；此时缓存只查找、不淘汰。
  for (std::int64_t slot = 0; slot < query_count; ++slot) {
    const auto row_id = row_ids[slot];
    if (row_id < 0 || row_id >= table.rows) {
      throw std::out_of_range(
          "Engram SSD row " + std::to_string(row_id) +
          " is outside layer " + std::to_string(table.layer_id));
    }
    auto* weight_row = weights + slot * weight_bytes;
    auto* scale_row = scales + slot * scale_bytes;
    const auto* cached = find_cached_row(table.layer_id, row_id, false);
    if (cached != nullptr) {
      std::memcpy(weight_row, cached->data(), static_cast<std::size_t>(weight_bytes));
      std::memcpy(
          scale_row, cached->data() + weight_bytes, static_cast<std::size_t>(scale_bytes));
      continue;
    }
    requests.push_back(IoReadRequest{
        table.weight_fd, weight_row, weight_bytes,
        row_offset(table.weight_offset, row_id, weight_bytes)});
    requests.push_back(IoReadRequest{
        table.scale_fd, scale_row, scale_bytes,
        row_offset(table.scale_offset, row_id, scale_bytes)});
  }

  io_uring_reader_->read_all(requests);
  commit_cache_rows(table, row_ids, query_count, weights, scales);
}

void BatchRowReader::commit_cache_rows(
    const RowTableSpec& table,
    const std::int64_t* row_ids,
    std::int64_t query_count,
    const std::uint8_t* weights,
    const std::uint8_t* scales) {
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;
  // 所有读取已经完成，按查询顺序提交缓存，不按线程或 CQE 完成顺序提交。
  for (std::int64_t slot = 0; slot < query_count; ++slot) {
    const auto row_id = row_ids[slot];
    if (find_cached_row(table.layer_id, row_id) == nullptr) {
      cache_row(
          table.layer_id, row_id,
          weights + slot * weight_bytes, scales + slot * scale_bytes);
    }
  }
}

}  // namespace sglang_engram_ssd::query
