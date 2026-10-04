# Local MVCC Transaction Manager

一个纯本地的 MVCC（多版本并发控制）事务管理器，仅使用 Python 标准库实现。
所有记录版本、事务状态、提交时间戳和日志只保存在**本地内存或本地文件**中，
不依赖 MySQL、PostgreSQL、Redis 或任何外部服务。

## 目录结构

```
mvcc/
  __init__.py   # 包导出
  errors.py     # MVCCError / TransactionStateError / WriteConflictError
  store.py      # MVCCStore + Transaction：版本链、快照、冲突检测、GC
  wal.py        # 本地文件 write-ahead log（JSON Lines，可选）
tests/
  test_mvcc.py  # 全部自动化测试（unittest，终端直接运行）
run_tests.sh    # 测试入口脚本
```

## 版本模型

- 每条逻辑记录（字符串 key）对应一条**版本链**，按提交时间戳升序排列。
- 每个版本包含：
  - `commit_ts`：创建该版本的事务的提交时间戳（全局单调递增计数器）；
  - `value`：记录值，或 `TOMBSTONE` 哨兵（表示该版本是一次删除）；
  - `tx_id`：创建该版本的事务 ID。
- 事务的未提交写入只保存在自己的 write set 中（内存），commit 时才
  分配提交时间戳并追加到版本链，同时（可选）写入本地 WAL 文件。
- 删除是**版本化 tombstone**：写入一个 `TOMBSTONE` 版本，旧版本保留在链上，
  因此旧 snapshot 依然能按可见性规则读到历史数据。

## 可见性规则（Snapshot Isolation）

- `begin()` 时事务固定一个 snapshot 时间戳 `S`（当前最新提交时间戳）。
- 读规则：对某个 key，取版本链中 `commit_ts <= S` 的最新版本；
  若该版本是 tombstone 或不存在，则读结果为 `None`。
- 事务**永远看不到**自己 snapshot 之后提交的数据（可重复读）；
  事务**总是看到**自己的未提交写入（read-your-writes），其他事务看不到。
- 不同时间点 begin 的事务看到各自时间点的数据，互不影响。

## 冲突规则（write-write conflict）

- 采用 **first-committer-wins**：若事务要写的 key 上已存在
  `commit_ts > S` 的版本（即 snapshot 之后有别的事务提交过该 key），
  则判定为写-写冲突。
- 冲突在**写操作时 fail-fast** 检测，并在 **commit 时再次校验**；
  冲突事务被中止并抛出 `WriteConflictError`，绝不会静默成功。
- 两个并发事务修改同一记录时，最多只有一个能提交成功。
- 写不同 key 的并发事务互不冲突，可以都提交。

## 垃圾回收（GC）

- GC 水位线 = 所有活动事务中最小的 snapshot 时间戳
  （无活动事务时为最新提交时间戳）。
- 对每条版本链：保留水位线可见的最新版本及所有更新版本，
  更早的版本已不可能被任何现在或将来的事务读取，安全删除。
- 若水位线可见的版本是 tombstone 且没有更新的版本，整条 key 直接移除。
- 长事务会"钉住"自己的 snapshot：它仍可读的历史版本不会被回收，
  直到该事务结束。

## API 一览

```python
from mvcc import MVCCStore, WriteConflictError

store = MVCCStore()                    # 纯内存
store = MVCCStore(wal_path="mvcc.log") # 内存 + 本地文件 WAL，重启可恢复

txn = store.begin()
txn.write("user:1", {"name": "alice"})
txn.read("user:1")
txn.delete("user:1")
txn.commit()                           # 冲突时抛 WriteConflictError 并中止
txn.rollback()                         # 丢弃全部未提交写入

store.gc()                             # 回收不再可读的旧版本
store.close()
```

## 运行测试

```sh
./run_tests.sh
# 或
python3 -m unittest discover -s tests -v
```

测试直接在终端执行并打印每个用例结果，不需要数据库或 GUI。覆盖场景：

- 不同时间点读取（snapshot 各自固定、互不可见新提交）；
- 快照内可重复读、read-your-writes、未提交数据对他人不可见；
- 并发写-写冲突（多线程竞争同一 key，恰好一个提交成功）；
- rollback 丢弃写入、事务结束后操作报错；
- 删除可见性（tombstone、旧 snapshot 仍可读、删除后重建）；
- 长事务钉住 snapshot 并阻止 GC 回收其可读版本；
- GC 回收旧版本 / tombstone key、尊重最老活动 snapshot；
- 大量随机事务（随机 begin/read/write/delete/commit/rollback，
  对照模型校验快照不变式）；
- WAL 持久化：提交日志写入本地文件，重开后恢复数据与时间戳。
