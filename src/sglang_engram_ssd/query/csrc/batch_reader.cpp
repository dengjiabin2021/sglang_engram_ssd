// 同步读取 checkpoint 行字节；类声明在 batch_reader.h，Python 绑定在 bindings.cpp。
#include "batch_reader.h"

#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <sys/types.h>

namespace sglang_engram_ssd::query {

BatchRowReader::BatchRowReader(
    const std::vector<RowTableSpec>& tables,
    std::int64_t io_uring_queue_depth) {
  if (io_uring_queue_depth <= 0) {
    throw std::invalid_argument("io_uring_queue_depth must be positive");
  }
  // 初始化时只复制元数据并借用 fd；空列表仍初始化 ring，供模型加载前检查。
  tables_.reserve(tables.size());
  for (const auto& table : tables) {
    if (!tables_.emplace(table.layer_id, table).second) {
      throw std::invalid_argument(
          "Duplicate Engram SSD layer: " + std::to_string(table.layer_id));
    }
  }
  io_uring_reader_ = std::make_unique<IoUringReader>(io_uring_queue_depth);
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

  // 返回 Tensor 句柄，不复制底层数据；工作区由调用方持有并在 H2D 完成后复用。
  return {std::move(weight), std::move(scale)};
}

void BatchRowReader::read_rows(
    std::int64_t layer_id,
    const at::Tensor& row_ids,
    at::Tensor& weight_out,
    at::Tensor& scale_out) {
  const RowTableSpec& table = open_table(layer_id);
  check_inputs(table, row_ids, weight_out, scale_out);

  const auto query_count = row_ids.size(0);
  if (query_count == 0) {
    return;  // 空批次不获取数据指针，也不发起 I/O。
  }

  // C++ 直接写 CPU 工作区，每个输入位置仍占一个槽位；不在 reader 中去重。
  read_rows_uring(
      table,
      row_ids.data_ptr<std::int64_t>(),
      nullptr,
      query_count,
      weight_out.data_ptr<std::uint8_t>(),
      scale_out.data_ptr<std::uint8_t>(),
      weight_out.stride(0),
      scale_out.stride(0));
}

void BatchRowReader::read_rows_to_slots(
    std::int64_t layer_id,
    const at::Tensor& row_ids,
    const at::Tensor& slot_ids,
    at::Tensor& weight_base,
    at::Tensor& scale_base) {
  const auto& table = open_table(layer_id);
  check_inputs(table, row_ids, weight_base, scale_base);
  TORCH_CHECK(
      slot_ids.defined() && slot_ids.device().is_cpu() &&
          slot_ids.scalar_type() == c10::ScalarType::Long &&
          slot_ids.dim() == 1 && slot_ids.is_contiguous(),
      "slot_ids must be contiguous CPU int64 [Q]");
  TORCH_CHECK(slot_ids.size(0) == row_ids.size(0), "row_ids and slot_ids must have equal length");
  std::unordered_set<std::int64_t> seen;
  seen.reserve(static_cast<std::size_t>(slot_ids.numel()));
  const auto* slots = slot_ids.data_ptr<std::int64_t>();
  for (std::int64_t i = 0; i < slot_ids.numel(); ++i) {
    if (slots[i] < 0 || slots[i] >= weight_base.size(0)) {
      throw std::out_of_range("Engram SSD target slot is outside output capacity");
    }
    if (!seen.insert(slots[i]).second) {
      throw std::invalid_argument("Engram SSD target slots must be unique");
    }
  }
  if (row_ids.numel() == 0) {
    return;
  }
  read_rows_uring(
      table, row_ids.data_ptr<std::int64_t>(), slots, row_ids.numel(),
      weight_base.data_ptr<std::uint8_t>(), scale_base.data_ptr<std::uint8_t>(),
      weight_base.stride(0), scale_base.stride(0));
}

void BatchRowReader::close() {
  // read_rows 返回前已收齐在途请求；调用方须先结束读取，不关闭借用的模型 fd。
  io_uring_reader_.reset();
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

void BatchRowReader::read_rows_uring(
    const RowTableSpec& table,
    const std::int64_t* row_ids,
    const std::int64_t* slot_ids,
    std::int64_t query_count,
    std::uint8_t* weights,
    std::uint8_t* scales,
    std::int64_t weight_stride,
    std::int64_t scale_stride) {
  const auto weight_bytes = table.dim;
  const auto scale_bytes = table.dim / 32;
  std::vector<IoReadRequest> requests;
  requests.reserve(static_cast<std::size_t>(query_count) * 2);

  // 先验证全部行并构造描述符，避免参数错误时留下访问输出内存的在途请求。
  for (std::int64_t query = 0; query < query_count; ++query) {
    const auto row_id = row_ids[query];
    if (row_id < 0 || row_id >= table.rows) {
      throw std::out_of_range(
          "Engram SSD row " + std::to_string(row_id) +
          " is outside layer " + std::to_string(table.layer_id));
    }
    const auto slot = slot_ids == nullptr ? query : slot_ids[query];
    requests.push_back(IoReadRequest{
        table.weight_fd, weights + slot * weight_stride, weight_bytes,
        row_offset(table.weight_offset, row_id, weight_bytes)});
    requests.push_back(IoReadRequest{
        table.scale_fd, scales + slot * scale_stride, scale_bytes,
        row_offset(table.scale_offset, row_id, scale_bytes)});
  }

  io_uring_reader_->read_all(requests);
}

}  // namespace sglang_engram_ssd::query
