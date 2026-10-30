// Python 绑定只负责参数转换和 GIL 管理，读取逻辑保留在 BatchRowReader 类中。
#include <torch/extension.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "batch_reader.h"

namespace py = pybind11;
using sglang_engram_ssd::query::BatchRowReader;
using sglang_engram_ssd::query::RowTableSpec;

PYBIND11_MODULE(_C, module) {
  module.doc() = "Engram SSD synchronous batch row reader";

  py::class_<BatchRowReader>(module, "BatchRowReader")
      .def(
          py::init([](const py::sequence& tables,
                      std::int64_t cache_bytes,
                      std::optional<std::int64_t> num_threads,
                      std::optional<std::int64_t> thread_pool_queue_depth,
                      std::optional<std::int64_t> io_uring_queue_depth,
                      const std::string& io_backend) {
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
                specs, cache_bytes, num_threads, thread_pool_queue_depth,
                io_uring_queue_depth, io_backend);
          }),
          py::arg("tables"),
          py::kw_only(),
          py::arg("cache_bytes"),
          py::arg("num_threads") = py::none(),
          py::arg("thread_pool_queue_depth") = py::none(),
          py::arg("io_uring_queue_depth") = py::none(),
          py::arg("io_backend") = "pread")
      .def(
          "allocate_workspace",
          &BatchRowReader::allocate_workspace,
          py::kw_only(),
          py::arg("layer_id"),
          py::arg("capacity"))
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
}
