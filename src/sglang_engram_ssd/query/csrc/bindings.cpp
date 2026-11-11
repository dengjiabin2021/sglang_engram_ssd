// Python 绑定只负责参数转换和 GIL 管理，读取逻辑保留在 BatchRowReader 类中。
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "batch_reader.h"
#include "prefetch_row_pool.h"

namespace py = pybind11;
using sglang_engram_ssd::query::BatchRowReader;
using sglang_engram_ssd::query::RowTableSpec;
using sglang_engram_ssd::query::PrefetchRowHandle;
using sglang_engram_ssd::query::PrefetchRowPool;
using sglang_engram_ssd::query::PrefetchRowPoolStats;
using sglang_engram_ssd::query::PrefetchRowResolution;

PYBIND11_MODULE(_C, module) {
  module.doc() = "Engram SSD synchronous row reader and pinned prefetch pool";

  py::class_<BatchRowReader>(module, "BatchRowReader")
      .def(
          py::init([](const py::sequence& tables,
                      std::int64_t io_uring_queue_depth) {
            // 这里只在构造时转换一次元数据。访问 Python 属性期间必须持有 GIL。
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
      .def_property_readonly("missing_slot_ids", &PrefetchRowHandle::missing_slot_ids);

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
          py::kw_only(), py::arg("handle"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "cancel", &PrefetchRowPool::cancel,
          py::kw_only(), py::arg("handle"),
          py::call_guard<py::gil_scoped_release>())
      .def("stats", [](const PrefetchRowPool& pool) {
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
