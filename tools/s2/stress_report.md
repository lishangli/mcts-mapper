# S2 大规模 DFG 压力扫描

生成器 `gen_dag.py`（自带独立求值器给出期望值），驱动 `stress.py`（串行调 `run_dfg.sh`），
判定复用 `regression.steady()` 与 `bridge_hint()`。本轮**不改动被测代码**（`mcts_bridge.py` /
`sched.py` / `run_dfg.sh` 均未修改），只做生成、驱动与测量。

## 资源上限（实测）

| 资源 | 数量 | 来源 |
|---|---|---|
| PE | 128 | `sched.PES` |
| 每 PE 每 operand 的 `io_in` 端口 | **4**（126 个 PE；2 个 PE 为 3） | `PES[i].mux.rdu_io_in_0/1.srcs` |
| op0 / op1 端口组是否重叠 | 否（128 个 PE 全不重叠） | 同上 |
| GIB | 153 | `netlist_gibs.json` |
| GIB mux 输出总数 | **1668** | `sum(len(g['mux']))` |
| IOB bank | **32**（常量流与 capture 共用） | `pick_iob` / `iob_stream_use` 的 `range(32)` |

## 指标

| tag | 形状 | objects | compute | edges | cin rows | IOB banks | streams | captures | remap | occ peak | expect | got | time | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chain12o2c3（stress 独立跑） | chain 12 | 19 | 12 | 26 | 202 | 3 | 1 | 2 | 4 | 224 | [4, 6] | [4, 6] | 527.1s | PASS |
| chain12o2c3（回归复测） | chain 12 | 19 | 12 | 26 | 208 | 3 | 1 | 2 | 1 | 244 | [4, 6] | [4, 6] | ~5 min | PASS |
| chain20o3c3 | chain 20 | 28 | 20 | 43 | - | - | 0 | - | 10（预算 12） | - | [8, 4, 6] | - | 1200s 超时 | **FAIL** |
| l4w5o3c3 | layered 4x5 | 28 | 20 | 43 | - | - | 0 | - | 10（预算 12） | - | [6, 6, 4] | - | 1200s 超时 | **FAIL** |

参考：已有 11 节点单输出用例 `dfg_big` 的 occ peak 为 141~185，首轮成功率 5/5（P0 放宽后）。
12 节点链式两次运行的 remap 分别是 4 和 1，说明它在 12 次预算内可稳定通过，但已不是「首轮即过」。

## 结论

1. **IOB bank 不是瓶颈。** 通过的最大配置只用了 3/32 个 bank（2 个 capture + 1 条常量流），
   且 `popcount(iob_ens) == captures + streams` 交叉校验一致。要触到 32 上限需要 ~30 个输出通道，
   在当前规模下远未到。
2. **GIB mux 容量不是瓶颈。** chain12 的 occ peak 为 224，仅占 1668 的 13%；按每节点约 18 个
   mux 外推，20 节点约需 370（22%），仍有大量余量。
3. **真正的瓶颈是 mod-4 相位约束叠加每 operand 4 个 pin 的端口组。** 20 节点无论链式还是网格，
   12 次 remap 内全部失败，原因清一色是 `no parity route`（少数 `no route`），而不是
   `no const IOB` / `no capture IOB` / `conflict`。`port_used` 按 PE 独占端口，一个 PE 上放两个
   节点（slot 0/1）时就只剩 4+4 个 pin 可用，`route_parity` 还必须在避开已占资源的简单路径上
   凑出精确的 `want = (c0 - ctx) % 4`，两者叠加使深度链的中后段先撞墙。
4. **12 节点链式是当前可行边界。** chain12 两次运行分别用了 4 次和 1 次 remap（预算 12），在预算
   内可稳定通过，但已不是首轮即过；一跳到 20 节点（两种形状）就退化为不可达。回归套件的第 9 例
   用的就是这个配置。

## 后续可做的两项（本轮未实施，需另立项）

- **c0 相位松弛**：`no parity route` 时先把该节点的 `c0` 抬 1~3（改变 `want`）再重试，而不是直接
  assert 掉整轮 placement。当前 `c0 = max(ctx[tail] + hs)` 取最小值，相位自由度被锁死。
- **端口组视角的 placement 奖励**：把「同一 PE 上两个节点的 operand pin 需求是否超过 4」纳入
  MCTS 的合法/惩罚项，避免把多个 2 输入节点压到同一 PE。

## 复现

```bash
cd /home/lsl/mapping/cgra-sw/s2

# 通过配置（12 节点链式、2 输出）
python stress.py --only chain12

# 已知不可达配置（20 节点，链式/网格）
python gen_dag.py --shape chain --layers 20 --width 1 --outputs 3 --consts 3 \
  --name chain20o3c3
./run_dfg.sh dfg_dag_chain20o3c3.json chain20o3c3

# 任意随机规模
python gen_dag.py --layers 4 --width 5 --outputs 3 --consts 3 --name l4w5o3
python gen_dag.py --shape chain --layers 16 --width 1 --outputs 3 --name chain16o3c3
```

`stress.py` 单配置约 2~9 分钟（取决于 remap 次数），全部串行执行：`run_dfg.sh` 会就地重写
`main_s2.c` / `cgra_execute.c`，并行会互相覆盖。
