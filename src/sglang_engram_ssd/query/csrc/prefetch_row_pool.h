// 调度级预取使用的 pinned 行池。实现在 prefetch_row_pool.cpp，Python 绑定在 bindings.cpp。
//
// 用途：后台预取线程把 SSD 中的 Engram 行读进池里，batch 组装时将命中行拷进暂存区。
// 池按 (layer_id,row_id) 建索引，同一行只占一个槽位、只读一次（跨请求、跨段去重）。
//
// 内存：两张 CPU pinned uint8 表 weight[S,D]、scale[S,D/32]，同一槽号在两张表中指同一行。
// 两张表都映射到 GPU 地址空间，engram_gather 可用槽号直接读取，不经过显式 H2D。
//
// 槽位状态：
//   FREE    空闲，在空闲链表中，不在索引中。
//   PENDING 已分配并登记索引，数据尚未读完；GPU 不得读取，也不得回收。
//   READY   数据已读完，可被 forward 命中。
//
// 引用计数与 LRU：
//   - 每个句柄对其持有的每个槽恰好贡献一次引用。
//   - 引用计数 > 0 的槽不会被回收或覆盖。
//   - READY 且引用计数为 0 的槽留在 LRU 链表中作为缓存；空闲槽不够时从 LRU 尾部回收。
//   - 零引用 PENDING 默认保留到 mark_ready。调用方确认没有在途写入时可显式回收。
//
// 调用方与线程：
//   - 后台预取线程：查询 PENDING 槽 → BatchRowReader::read_rows_to_slots → mark_ready。
//   - 控制器在准备普通段时 acquire；调度判断 is_ready，消费完归还引用。
//   - 组装 batch 时 resolve 取得 READY 槽的引用，完成暂存区复制之后 release。
// 可变状态在内部互斥锁下维护，固定容量／步长只读无锁，可以从上述线程并发调用。
// runtime 创建并持有池，先停止 I/O 和 GPU 使用并归还引用，再调用 close 释放行池。
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

// 一组槽位引用。由 acquire 或 resolve 创建，必须且只能通过 release 或 cancel 归还一次。
// 析构不会自动归还引用：Python 侧丢弃句柄而不 release，会让这些槽永久无法回收。
// 句柄内保存池的 shared_ptr，池对象至少存活到最后一个句柄析构。
// 引用字段仅由池在 mutex_ 下修改；只读缺失行元数据发布后不变，调用方不得原地修改。
class PrefetchRowHandle {
 public:
  PrefetchRowHandle(const PrefetchRowHandle&) = delete;
  PrefetchRowHandle& operator=(const PrefetchRowHandle&) = delete;

  // acquire 新建、需要调用方从 SSD 读取的行，两个数组一一对应，均为 CPU int64 [M]。
  // missing_row_ids()[i] 是全局行号，missing_slot_ids()[i] 是它要写入的槽号。
  // resolve 返回的句柄这两个数组为空。数组只读；句柄 release/cancel 后不得再用于 I/O。
  at::Tensor missing_row_ids() const;
  at::Tensor missing_slot_ids() const;

  // 无输入。acquire 返回的连续 CPU int64 [U] 全局行号和槽号，按首次出现顺序去重。
  // resolve 句柄的两个数组为空。
  // 数组借用句柄的只读元数据，槽地址仅在句柄持有引用期间有效。
  at::Tensor row_ids() const;
  at::Tensor slot_ids() const;

 private:
  friend class PrefetchRowPool;
  // 输入 owner 为创建句柄的池；输出为空句柄并持有 owner 的共享所有权，不取得任何槽引用。
  // 仅 PrefetchRowPool 可构造，随后在池锁下填入槽信息；不得据此直接写入行池。
  explicit PrefetchRowHandle(std::shared_ptr<PrefetchRowPool> owner);

  std::shared_ptr<PrefetchRowPool> owner_;  // 保证池对象至少与句柄同寿命，不代替 release。
  std::vector<std::int64_t> slots_;          // 本句柄持有引用的全部槽（复用的和新建的），无重复。
  std::vector<std::int64_t> created_slots_;  // 其中由本句柄新建为 PENDING 的槽，cancel 只撤销这些。
  at::Tensor missing_row_ids_;
  at::Tensor missing_slot_ids_;
  at::Tensor row_ids_;
  at::Tensor slot_ids_;
  bool released_ = false;                    // 已 release 或 cancel；在池锁下读写。
};

// resolve 的结果：按输入顺序给出每个查询位置对应的槽号。
// 由池构造后交给 batch 组装方，元数据只读；handle 在槽数据复制到暂存区后显式归还。
struct PrefetchRowResolution {
  at::Tensor slot_ids;  // CPU int64 [Q]；命中为槽号，未命中为 -1。重复行号得到相同槽号。
  at::Tensor hit_mask;  // CPU bool [Q]；只有 READY 才算命中，PENDING 和不存在都算未命中。
  std::shared_ptr<PrefetchRowHandle> handle;  // 对每个命中的槽持有一次临时引用。
};

// stats() 的快照。占用类字段总是有效；累计类字段只在 track_stats=true 时增长。
// 池在锁内复制，调用方拥有副本并可无锁读取；不持有行数据或槽引用。
struct PrefetchRowPoolStats {
  // 占用类：当前时刻的状态。
  std::int64_t capacity = 0;          // 槽位总数 S。
  std::int64_t payload_bytes = 0;     // S * (D + D/32)，不含对齐余量和元数据。
  std::int64_t occupied_slots = 0;    // 非 FREE 的槽数（PENDING + READY）。
  std::int64_t referenced_slots = 0;  // 引用计数 > 0 的槽数。
  std::int64_t pending_slots = 0;
  std::int64_t ready_slots = 0;
  std::int64_t cached_slots = 0;      // 在 LRU 中的槽数，即 READY 且无引用。
  // 累计类：自创建以来的总数。
  std::int64_t submitted_rows = 0;    // 实际交给 reader 读取的行数，由调用方 record_submitted 登记。
  std::int64_t dedup_saved_rows = 0;  // acquire 输入行数减去新建行数：批内重复加上池中已有的行。
  std::int64_t forward_hits = 0;      // 批次组装时 READY 命中的查询数。
  std::int64_t forward_misses = 0;    // 批次组装时缺失的查询数。
  double miss_read_ms = 0;            // 同步补读的累计耗时。
  std::int64_t deferred_requests = 0; // 调度时被暂缓的请求次数。
  std::int64_t evicted_slots = 0;     // 从 LRU 淘汰的 READY 槽数。
  std::int64_t evicted_unused_slots = 0;  // 上述槽中，自 READY 起从未被 acquire 或 resolve 命中的数量。
  bool tracking_enabled = false;
  bool closed = false;
};

// 线程、布局、状态与生命周期见文件开头；本类只管理 CPU pinned 行，不定位文件或执行 I/O。
class PrefetchRowPool : public std::enable_shared_from_this<PrefetchRowPool> {
 public:
  // 分配两张 CPU pinned 表并取得它们的 GPU 映射地址，失败直接抛异常。
  // 槽位数 S = pool_bytes / (D + D/32)；pool_bytes 只计行数据，不含对齐余量、槽元数据和索引。
  // 两张表的起始地址都按 4096 字节对齐，表内按行连续存放。
  // 输入 dim 为正数 D 且是 32 的倍数；pool_bytes 至少容纳一行；track_stats 只控制累计统计。
  // 必须由 std::shared_ptr 管理（pybind 的 holder 即是）：acquire/resolve 用 shared_from_this() 把池绑定到句柄上。
  PrefetchRowPool(std::int64_t dim, std::int64_t pool_bytes, bool track_stats = false);
  // 无输入输出；析构释放成员所有权。正常退出应先显式 close 校验没有活跃引用或在途写入。
  ~PrefetchRowPool() = default;
  PrefetchRowPool(const PrefetchRowPool&) = delete;
  PrefetchRowPool& operator=(const PrefetchRowPool&) = delete;

  // 两张表的 CPU 视图（uint8 [S,D] 和 [S,D/32]），以及 GPU 可直接访问的起始地址。
  // CPU 侧只允许写入预留的 PENDING 槽；GPU 侧只允许读取已发布且仍持有引用的槽。
  // 下列 getter 均无输入；前四项在池关闭后抛异常，返回 Tensor／指针不额外增加槽引用。
  at::Tensor weight() const;
  at::Tensor scale() const;
  std::uintptr_t weight_device_ptr() const;
  std::uintptr_t scale_device_ptr() const;
  std::int64_t capacity() const;  // 固定槽容量 S，可在关闭后查询，不代表当前可回收槽数。
  std::int64_t weight_stride() const;  // 每行 weight 字节数，等于 D。
  std::int64_t scale_stride() const;   // 每行 scale 字节数，等于 D/32。

  // 为一批行登记引用，不做 I/O。layer_id 非负；row_ids 为连续 CPU int64 [Q] 非负全局行号，可重复。
  //   - 已在池中的行（PENDING 或 READY）：复用原槽，引用 +1。
  //   - 不在池中的行：分配新槽并置为 PENDING，列在返回句柄的 missing_row_ids/missing_slot_ids 中，
  //     由调用方读取后调用 mark_ready。
  // 空间不足时返回 nullptr，池状态不变。
  // 若某行处于被 cancel 后的失败状态，抛异常。
  std::shared_ptr<PrefetchRowHandle> acquire(std::int64_t layer_id, const at::Tensor& row_ids);

  // 无输入；同步返回空闲槽与零引用 READY 槽之和，不增加引用，池关闭时抛异常。
  std::int64_t available_slots() const;

  // 输入非负 layer_id 与连续 CPU int64 [Q] 非负全局行号，可重复。
  // 返回去重后尚无槽或位于 LRU 的行数；limit 非负时，计数超过 limit 即返回。
  // 不增加引用，不读盘，池关闭时抛异常。
  std::int64_t slots_needed(std::int64_t layer_id, const at::Tensor& row_ids, std::int64_t limit = -1) const;

  // 输入非负 layer_id 与连续 CPU int64 [Q] 非负全局行号，可重复。
  // 同步返回按输入位置对齐的连续 CPU int64 [Q] 槽号，非 PENDING 为 -1，失败槽抛异常。
  // 不加引用、不读盘；调用方须自行持有槽引用并串行化读写后才能使用这些槽号。
  // 普通段之间共享已占槽的行，分块读前用此方法跳过其它段已发布的行。
  at::Tensor pending_slots(std::int64_t layer_id, const at::Tensor& row_ids) const;

  // 输入 handle 须属于本池。持有的所有槽都已 READY 时返回 true，空句柄返回 true；不等待 I/O。
  // 句柄已 release/cancel，或其中某槽处于失败状态时，抛异常。
  bool is_ready(const std::shared_ptr<PrefetchRowHandle>& handle) const;

  // 把一批 PENDING 槽置为 READY。slot_ids 为连续 CPU int64 [Q]，范围 [0,S)，允许重复。
  // 必须在对应的 read_rows_to_slots 返回、数据已写完之后调用。
  // 先检查整批：任一槽不是 PENDING 或处于失败状态就抛异常，且不修改任何槽。
  // 开启统计时从本次发布开始跟踪 READY 命中。
  void mark_ready(const at::Tensor& slot_ids);

  // batch 组装查询：layer_id 非负，row_ids 为连续 CPU int64 [Q] 非负行号，可重复。
  // 输出 PrefetchRowResolution 按原顺序保存槽号、命中掩码及独立临时句柄，不执行 I/O。
  // 只有 READY 算命中。每个命中的槽（去重后）在返回的句柄中持有一次引用，
  // 保证复制期间槽不被回收；调用方读完槽数据后 release 该句柄。
  PrefetchRowResolution resolve(std::int64_t layer_id, const at::Tensor& row_ids);

  // 输入 handle 为本池句柄；无返回值，归还其全部引用，对同一句柄重复调用无效果。
  // 默认引用归零的 PENDING 保留至 mark_ready。discard_pending 为 true 时回收归零的 PENDING，
  // 其它句柄共享的槽仍可继续读取。使用此选项时须确保每个在途写入由仍存活的引用保护。
  void release(const std::shared_ptr<PrefetchRowHandle>& handle, bool discard_pending = false);

  // 输入 handle 为本池句柄；无返回值，归还引用并撤销其新建且仍为 PENDING 的槽。
  // 只能在这些槽的 I/O 没有开始、或已全部结束（CQE 收齐）后调用，否则 I/O 可能写入已回收的槽。
  //   - 撤销的槽若没有其他引用，立即回收为 FREE。
  //   - 若其他句柄也引用了它，标记为失败：其他句柄的 is_ready 会抛异常，最后一个引用归还时回收。
  // 句柄中复用的 READY 槽和别人新建的 PENDING 槽只减引用。对同一句柄重复调用无效果。
  void cancel(const std::shared_ptr<PrefetchRowHandle>& handle);

  // 无输入；在锁内复制并返回一份统计快照，字段含义见 PrefetchRowPoolStats，不改变计数。
  PrefetchRowPoolStats stats() const;

  // 以下三个函数输入为非负计数／有限毫秒值，无返回值；仅 track_stats=true 时校验并累加。
  void record_submitted(std::int64_t rows);  // 实际交给 reader 读取的行数。
  void record_forward(std::int64_t hits, std::int64_t misses, double miss_ms);  // 一次 forward 的命中/补读。
  void record_deferred(std::int64_t count);  // 一次调度中被暂缓的请求数。

  // 释放两张 pinned 表和全部索引。要求所有句柄都已归还且没有 PENDING 槽，否则抛异常。
  // 调用方应先停止后台提交、等 I/O 和 GPU 使用结束。成功关闭后重复调用无效果。不涉及模型 fd。
  void close();

 private:
  // 池锁保护的槽状态；FREE 可分配，PENDING 保留给在途写入，READY 才能被 GPU 消费。
  // 状态随槽存在，所有转换由池方法执行，调用方不能直接发布或覆盖槽。
  enum class State : std::uint8_t { FREE, PENDING, READY };
  // 一个层内全局行的身份；随索引／槽复制，不持有外部资源，发布后只在池锁下更新。
  struct RowKey {
    std::int64_t layer_id = -1;
    std::int64_t row_id = -1;
    // 输入另一行键；返回层号和全局行号均相同的结果，不修改任一键。
    bool operator==(const RowKey&) const = default;
  };
  // 无状态的索引哈希器，由 unordered_map 持有，可并发只读调用，不管理池生命周期。
  struct RowKeyHash {
    // 输入借用的行键，返回结合层号与全局行号的哈希值，不修改键或池状态。
    std::size_t operator()(const RowKey& key) const;
  };
  // 槽元数据。previous/next 复用为两条链表的指针：
  //   FREE 槽：只用 next，串成空闲单链表（free_head_ 起）。
  //   READY 且无引用的槽：previous/next 串成 LRU 双链表（lru_head_ 最近使用，lru_tail_ 最久未用）。
  //   其他槽：两个指针都为 -1。
  // 池创建并持有元数据到 close，所有字段仅在池锁下修改，不持有行数据或模型 fd。
  struct Slot {
    RowKey key;                    // 非 FREE 时有效，与 index_ 中的键一致。
    std::int64_t references = 0;   // 持有该槽的句柄数。
    std::int64_t previous = -1;
    std::int64_t next = -1;
    State state = State::FREE;
    bool failed = false;           // 新建它的 acquire 被 cancel，但仍有其他句柄引用；不可再命中或复用。
    bool used = false;             // 开启统计时，记录本次 READY 发布后是否命中过。
  };

  // 以下辅助函数都要求调用方已持有 mutex_，内部不做 I/O，也不访问 Python 对象。
  // 无输入或输出；已关闭时抛异常，不改变状态。
  void require_open() const;
  // 输入 handle；不返回值或改状态，句柄为空／不属本池时抛异常。
  void require_handle(const std::shared_ptr<PrefetchRowHandle>& handle) const;
  // 输入借用 slot；不返回值或改状态，槽已标为失败时抛异常。
  void require_healthy(const Slot& slot) const;
  // 输入位于 LRU 的 slot_id；无返回值，将该槽摘出链表并更新缓存槽计数。
  void remove_lru(std::int64_t slot_id);
  // 输入零引用 READY 的 slot_id；无返回值，将槽插入 LRU 头部并更新缓存槽计数。
  void add_lru(std::int64_t slot_id);
  // 输入健康且非 FREE 的 slot_id；无返回值，引用 +1，零引用 READY 从 LRU 移出。
  // 开启统计时，同时标记 acquire 或 resolve 对 READY 槽的命中。
  void pin(std::int64_t slot_id);
  // 输入持有引用的 slot_id 及是否回收未读槽；无返回值，引用 -1，归零时 READY 进 LRU。
  // 失败槽或允许丢弃的 PENDING 槽归零时回收。
  void unpin(std::int64_t slot_id, bool discard_pending);
  // 输入可回收且零引用的 slot_id；无返回值，删除索引，更新状态及计数，放回空闲链表。
  void recycle(std::int64_t slot_id);
  // 无输入，返回并摘出一个 FREE 槽；无空闲槽时先回收 LRU 尾部，调用前须确认有可用空间。
  // 开启统计时，回收 LRU 尾部前累计淘汰和未命中淘汰次数。
  std::int64_t take_slot();
  // 输入借用的可变 handle 及是否回收未读槽；无返回值，逐槽 unpin 并标记句柄已归还。
  void release_slots(PrefetchRowHandle& handle, bool discard_pending = false);

  const std::int64_t capacity_;
  const std::int64_t weight_stride_;
  const std::int64_t scale_stride_;
  const bool track_stats_;
  at::Tensor weight_storage_;   // 实际分配的 pinned 内存（含对齐余量），持有所有权。
  at::Tensor scale_storage_;
  at::Tensor weight_;           // storage 中对齐后的 [S,D] 视图。
  at::Tensor scale_;            // storage 中对齐后的 [S,D/32] 视图。
  std::uintptr_t weight_device_ptr_ = 0;  // weight_ 起始地址在 GPU 侧的映射。
  std::uintptr_t scale_device_ptr_ = 0;
  mutable std::mutex mutex_;    // 保护以下全部可变状态。
  std::vector<Slot> slots_;
  std::unordered_map<RowKey, std::int64_t, RowKeyHash> index_;  // 非 FREE 槽：(layer,row) -> 槽号。
  std::int64_t free_head_ = -1;
  std::int64_t lru_head_ = -1;
  std::int64_t lru_tail_ = -1;
  // 以下计数与 slots_ 的实际状态保持一致，用于快速判断空间和输出统计。
  std::int64_t free_slots_ = 0;
  std::int64_t referenced_slots_ = 0;
  std::int64_t pending_slots_ = 0;
  std::int64_t ready_slots_ = 0;
  std::int64_t cached_slots_ = 0;
  PrefetchRowPoolStats counters_;  // 只使用其中的累计类字段。
  bool closed_ = false;
};

}  // namespace sglang_engram_ssd::query
