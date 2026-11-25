// Python 绑定：转发行读取器、行池，并提供 decode 用的 CUDA 流标志操作。
// 读取与行池重操作释放 GIL，返回或抛异常时恢复；参数转换与 Tensor 属性返回保留 GIL。
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <memory>
#include <vector>

#include "batch_reader.h"
#include "prefetch_row_pool.h"

namespace sglang_engram_ssd::query {
namespace {

// 输入 CUDA driver 返回码和调用名。成功时返回，失败时附带 driver 错误名抛异常。
void check_driver(CUresult result, const char* name) {
  if (result != CUDA_SUCCESS) {
    const char* error = nullptr;
    cuGetErrorName(result, &error);
    TORCH_CHECK(false, name, ": ", error, " (", static_cast<int>(result), ")");
  }
}

// 输入 CPU pinned int32 [1] 零标志和 CUDA stream，返回标志的 GPU 映射地址；不支持时返回 0。
// probe 为真时在空流上实测读写并收齐已提交操作；调用方持有 Tensor 的所有权。
std::uintptr_t stream_flag_address(const at::Tensor& flags, std::uintptr_t stream, bool probe) {
  void* mapped = nullptr;
  const auto status = cudaHostGetDevicePointer(&mapped, flags.data_ptr<std::int32_t>(), 0);
  if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
    cudaGetLastError();
    return 0;
  }
  TORCH_CHECK(status == cudaSuccess, "Engram cudaHostGetDevicePointer: ", cudaGetErrorString(status));
  const auto address = reinterpret_cast<CUdeviceptr>(mapped);
  if (!probe) {
    return address;
  }
  const auto cuda_stream = reinterpret_cast<CUstream>(stream);
  const auto write = cuStreamWriteValue32(cuda_stream, address, 0, CU_STREAM_WRITE_VALUE_DEFAULT);
  if (write == CUDA_ERROR_NOT_SUPPORTED || write == CUDA_ERROR_INVALID_VALUE) {
    return 0;
  }
  check_driver(write, "Engram cuStreamWriteValue32 probe");
  const auto wait = cuStreamWaitValue32(cuda_stream, address, 0, CU_STREAM_WAIT_VALUE_GEQ);
  check_driver(cuStreamSynchronize(cuda_stream), "Engram cuStreamSynchronize probe");
  if (wait == CUDA_ERROR_NOT_SUPPORTED || wait == CUDA_ERROR_INVALID_VALUE) {
    return 0;
  }
  check_driver(wait, "Engram cuStreamWaitValue32 probe");
  return address;
}

// 输入 CUDA stream、对齐的 GPU 映射地址和目标值，入队等待 *address >= value，立即返回。
// 调用方保持标志内存至流使用结束；CUDA 提交失败时抛异常。
void wait_stream_flag(std::uintptr_t stream, std::uintptr_t address, std::uint32_t value) {
  check_driver(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), address, value,
                                 CU_STREAM_WAIT_VALUE_GEQ), "Engram cuStreamWaitValue32");
}

// 输入 CUDA stream 和 CPU pinned int32 [1] 的 GPU 映射地址，按流顺序重置标志，立即返回。
// 必须在上一次 forward 使用该缓冲区结束后执行，最后由 hash D2H event 发布重置结果。
void reset_stream_flag(std::uintptr_t stream, std::uintptr_t address) {
  check_driver(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream), address, 0, CU_STREAM_WRITE_VALUE_DEFAULT),
               "Engram cuStreamWriteValue32");
}

// 输入 CPU pinned int32 [1]；暂存区写完后以 release 顺序放行 H2D，无返回值。
void set_stream_flag(const at::Tensor& flags) {
  std::atomic_ref<std::int32_t>(*flags.data_ptr<std::int32_t>()).store(1, std::memory_order_release);
}

}  // namespace
}  // namespace sglang_engram_ssd::query

namespace py = pybind11;
using sglang_engram_ssd::query::BatchRowReader;
using sglang_engram_ssd::query::RowTableSpec;
using sglang_engram_ssd::query::PrefetchRowHandle;
using sglang_engram_ssd::query::PrefetchRowPool;
using sglang_engram_ssd::query::PrefetchRowPoolStats;
using sglang_engram_ssd::query::PrefetchRowResolution;

// 输入为 Python 导入系统提供的 module；登记读取、行池和 CUDA 流标志接口，不创建运行资源。
PYBIND11_MODULE(_C, module) {
  module.doc() = "Engram SSD synchronous row reader and pinned prefetch pool";
  using namespace sglang_engram_ssd::query;
  module.def("stream_flag_address", &stream_flag_address, py::arg("flags"), py::arg("stream"), py::arg("probe"),
             py::call_guard<py::gil_scoped_release>());
  module.def("wait_stream_flag", &wait_stream_flag, py::arg("stream"), py::arg("address"), py::arg("value"),
             py::call_guard<py::gil_scoped_release>());
  module.def("reset_stream_flag", &reset_stream_flag, py::arg("stream"), py::arg("address"),
             py::call_guard<py::gil_scoped_release>());
  module.def("set_stream_flag", &set_stream_flag, py::arg("flags"));

  py::class_<BatchRowReader>(module, "BatchRowReader")
      .def(
          py::init([](const py::sequence& tables,
                      std::int64_t io_uring_queue_depth) {
            // 输入 tables 为 Python 表元数据序列，深度为该实例正数在途上限；返回独占 reader。
            // 这里只在构造时转换一次元数据。访问 Python 属性期间必须持有 GIL，FD 始终借用。
            std::vector<RowTableSpec> specs;
            specs.reserve(py::len(tables));
            for (const py::handle table : tables) {
              specs.push_back(RowTableSpec{
                  table.attr("layer_id").cast<std::int64_t>(),
                  table.attr("rows").cast<std::int64_t>(),
                  table.attr("dim").cast<std::int64_t>(),
                  table.attr("weight_fd").cast<int>(),
                  table.attr("scale_fd").cast<int>(),
                  table.attr("weight_offset").cast<std::int64_t>(),
                  table.attr("scale_offset").cast<std::int64_t>(),
              });
            }
            return std::make_unique<BatchRowReader>(
                specs, io_uring_queue_depth);
          }),
          py::arg("tables"),
          py::kw_only(),
          py::arg("io_uring_queue_depth") = 32)
      .def(
          "allocate_workspace",
          &BatchRowReader::allocate_workspace,
          py::kw_only(),
          py::arg("layer_id"),
          py::arg("capacity"))
      .def(
          "read_rows_to_slots",
          &BatchRowReader::read_rows_to_slots,
          py::kw_only(),
          py::arg("layer_id"),
          py::arg("row_ids"),
          py::arg("slot_ids"),
          py::arg("weight_base"),
          py::arg("scale_base"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "read_rows",
          &BatchRowReader::read_rows,
          py::kw_only(),
          py::arg("layer_id"),
          py::arg("row_ids"),
          py::arg("weight_out"),
          py::arg("scale_out"),
          // Tensor 参数先完成转换，再释放 GIL；返回或抛异常时自动恢复 GIL。
          py::call_guard<py::gil_scoped_release>())
      .def("close", &BatchRowReader::close);

  py::class_<PrefetchRowHandle, std::shared_ptr<PrefetchRowHandle>>(module, "PrefetchRowHandle")
      .def_property_readonly("missing_row_ids", &PrefetchRowHandle::missing_row_ids)
      .def_property_readonly("missing_slot_ids", &PrefetchRowHandle::missing_slot_ids)
      .def_property_readonly("row_ids", &PrefetchRowHandle::row_ids)
      .def_property_readonly("slot_ids", &PrefetchRowHandle::slot_ids);

  py::class_<PrefetchRowResolution>(module, "PrefetchRowResolution")
      .def_readonly("slot_ids", &PrefetchRowResolution::slot_ids)
      .def_readonly("hit_mask", &PrefetchRowResolution::hit_mask)
      .def_readonly("handle", &PrefetchRowResolution::handle);

  py::class_<PrefetchRowPool, std::shared_ptr<PrefetchRowPool>>(module, "PrefetchRowPool")
      .def(
          py::init<std::int64_t, std::int64_t, bool>(),
          py::kw_only(),
          py::arg("dim"),
          py::arg("pool_bytes"),
          py::arg("track_stats") = false)
      .def_property_readonly("weight", &PrefetchRowPool::weight)
      .def_property_readonly("scale", &PrefetchRowPool::scale)
      .def_property_readonly("weight_device_ptr", &PrefetchRowPool::weight_device_ptr)
      .def_property_readonly("scale_device_ptr", &PrefetchRowPool::scale_device_ptr)
      .def_property_readonly("capacity", &PrefetchRowPool::capacity)
      .def_property_readonly("weight_stride", &PrefetchRowPool::weight_stride)
      .def_property_readonly("scale_stride", &PrefetchRowPool::scale_stride)
      .def(
          "acquire", &PrefetchRowPool::acquire,
          py::kw_only(), py::arg("layer_id"), py::arg("row_ids"),
          py::call_guard<py::gil_scoped_release>())
      .def("available_slots", &PrefetchRowPool::available_slots, py::call_guard<py::gil_scoped_release>())
      .def(
          "slots_needed", &PrefetchRowPool::slots_needed,
          py::kw_only(), py::arg("layer_id"), py::arg("row_ids"), py::arg("limit") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "pending_slots", &PrefetchRowPool::pending_slots,
          py::kw_only(), py::arg("layer_id"), py::arg("row_ids"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "is_ready", &PrefetchRowPool::is_ready,
          py::kw_only(), py::arg("handle"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "mark_ready", &PrefetchRowPool::mark_ready,
          py::kw_only(), py::arg("slot_ids"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "resolve", &PrefetchRowPool::resolve,
          py::kw_only(), py::arg("layer_id"), py::arg("row_ids"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "release", &PrefetchRowPool::release,
          py::kw_only(), py::arg("handle"), py::arg("discard_pending") = false,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "cancel", &PrefetchRowPool::cancel,
          py::kw_only(), py::arg("handle"),
          py::call_guard<py::gil_scoped_release>())
      .def("stats", [](const PrefetchRowPool& pool) {
        // 输入为借用的 pool，输出为独立 Python 字典；先释放 GIL 取快照，再恢复 GIL 转换结果。
        PrefetchRowPoolStats snapshot;
        {
          py::gil_scoped_release release;
          snapshot = pool.stats();
        }
        // 在池锁和释放 GIL 的作用域之外构建 Python 对象。
        py::dict result;
        result["capacity"] = snapshot.capacity;
        result["payload_bytes"] = snapshot.payload_bytes;
        result["occupied_slots"] = snapshot.occupied_slots;
        result["referenced_slots"] = snapshot.referenced_slots;
        result["pending_slots"] = snapshot.pending_slots;
        result["ready_slots"] = snapshot.ready_slots;
        result["cached_slots"] = snapshot.cached_slots;
        result["submitted_rows"] = snapshot.submitted_rows;
        result["dedup_saved_rows"] = snapshot.dedup_saved_rows;
        result["forward_hits"] = snapshot.forward_hits;
        result["forward_misses"] = snapshot.forward_misses;
        result["miss_read_ms"] = snapshot.miss_read_ms;
        result["deferred_requests"] = snapshot.deferred_requests;
        result["evicted_slots"] = snapshot.evicted_slots;
        result["evicted_unused_slots"] = snapshot.evicted_unused_slots;
        result["tracking_enabled"] = snapshot.tracking_enabled;
        result["closed"] = snapshot.closed;
        return result;
      })
      .def(
          "record_submitted", &PrefetchRowPool::record_submitted,
          py::kw_only(), py::arg("rows"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "record_forward", &PrefetchRowPool::record_forward,
          py::kw_only(), py::arg("hits"), py::arg("misses"), py::arg("miss_ms"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "record_deferred", &PrefetchRowPool::record_deferred,
          py::kw_only(), py::arg("count"),
          py::call_guard<py::gil_scoped_release>())
      .def("close", &PrefetchRowPool::close, py::call_guard<py::gil_scoped_release>());
}
