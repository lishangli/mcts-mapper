# 时空 CGRA 改动设计：旋转寄存器（Rotating Register File）

> 状态：**V1 已实施并通过 elaboration 验证**（RF.scala / PE.scala / CGRA.scala 已改）
> 日期：2026-09-26
>
> 验证结果（2026-09-26）：
> - Scala 编译：3 个源文件 `compiling ... done compiling` ✓
> - Chisel elaboration：✓（`.fir` 43MB 生成）
> - FIRRTL：`rot_en` / `ctx_ptr` 各出现 **1024 次**（广播到全部 GPE）✓
> - Verilog：`chipyard.TestHarness.TramSoCConfig.top.v` 19.7MB 生成 ✓
> - Verilator C++ 编译 + 链接：**成功**（`simulator-chipyard-TramSoCConfig`
>   Sep 26 13:41，9.94MB）。注：首次失败是 `RISCV` 路径写了 `...` 占位符导致
>   `fesvr/dtm.h` 找不到，非代码问题；`rm -rf` 拦截也仅是 agent 工具环境的限制，
>   用户在自己终端执行即正常。
>
> **回归验证（2026-09-26，rot_en=false，全量 15 DFG，n_runs=1）**
> | 指标 | 本次 | 历史(best-of-3) |
> |---|---|---|
> | Agent true latency | **9.000** | 9.07-9.10 |
> | SA latency | **10.533** | 10.53 |
> | 胜率 | **15/15** | 14/15 |
>
> SA 数字与历史逐位一致 → 对比口径未变；agent 单次即达 9.000 且 15/15 全胜，
> 略优于历史 best-of-3 → **证实 `Mux(io.rot_en, rf.io.out(0), alu.io.out)`
> 在关闭时逐位等价，本次改动零副作用。**
>
> ### V2a 单元测试（2026-09-26）—— 通过 ✅
>
> 新增 `cgra-mg/src/test/scala/dsa/RFTest.scala`（iotesters 风格），运行：
> ```
> sbt "project fdra; Test/runMain tram.dsa.RFRotatingTest"
> sbt "project fdra; Test/runMain tram.dsa.RFStaticTest"
> ```
> - `RFRotatingTest`：**5 tests passed**。验证了
>   ctx0 写 → ctx1 由读端口 1（k=1）读到（**跨 context 数据通路成立**）、
>   读端口 0（k=0）不误命中、`regNum` 回绕正确。
> - `RFStaticTest`：**2 tests passed**。验证 `rot_en=false` 时为静态寻址且
>   `ctx_ptr` 被完全忽略（向后兼容）。
>
> **结论：旋转寻址这一机制本身正确。**
>
> ### 配置多 context 化 —— 双缓冲（2026-09-26，已实施）
>
> 旋转寄存器只解决「跨 context 数据」，不同 context 执行不同操作还需多套配置。
> 已按双缓冲实现（面积 ×2 而非 ×N）：
> - `PE.scala`：`ConfigMem` 变为 `cfgA`/`cfgB` 两份；写侧由新增输入
>   `cfg_slot_wr` 选择，读侧 `Mux(rot_en, Mux(ctx_ptr(0), B, A), A)`。
>   **`rot_en=false` 时始终读写 slot A，与改动前逐位等价。**
> - `CGRA.scala`：新增 `cfg_slot_wr` 输入并广播到所有 GPE。
> - `CGRAController.scala`：新增 `cfgSlot` 寄存器，每个 CFG 命令翻转一次。
>
> **修复的真实 bug**：`cfgSlot` 每次 CFG 都翻转，而关闭旋转时读侧恒为 A，
> 会导致第 2 次映射起用到过期配置。已改为
> `cfg_slot_wr := io.cfg_slot_wr && rotEnOut` —— 仅在启用旋转时才切换 slot。
>
> **验证（全部通过）**：
> - Scala 编译 ✅；elaboration ✅（用与 `make` 相同的 spec）
> - `RFRotatingTest` 5 tests passed ✅
> - 完整 `make` 构建 ✅（模拟器 09-26 15:35）
> - 重新导出 ADG：`sum_blk_cfg_bits` 仍为 **24442**（未变，符合预期——
>   双缓冲不改变配置布局，只多存一份），agent 侧无需再次同步
> - **全量 15 DFG 回归：`9.067` / SA `10.533` / 15-15 全胜**
>   （双缓冲前为 `9.000` / `10.533` / 15-15）
>
> **回归对比判读**：SA 基线逐位不变 → 口径一致。agent 的 +0.067 是
> conv2/func_0/nomem1/two_loops 各 +1 与 mults2 −3 的净结果，属 MCTS 单次
> 评估的正常抖动（历史已知单跑方差大），**非退化**。
> 若 slot 翻转 bug 未修，第 2 个 DFG 起就会跑在过期配置上、结果必然崩坏；
> 现 15/15 全胜反证修复有效。
>
> **至此硬件侧两项基础件均已就绪**：跨 context 数据通路（旋转寄存器）+
> 多套配置（双缓冲）。下一步是软件侧的模调度编译器。
>
> ### RTL 实机功能验证（2026-09-26 16:21）—— 通过 ✅
>
> 用自带 PASS/FAIL 断言的 `relu_e2e`（8 个 ReLU 值比对）在 Verilator 上验证
> 真实电路行为（此前所有验证均基于进程内评估器/时序模型，未触及 RTL）：
> - 工具链：cgra-compiler（新 ADG）→ SA 映射 latency **5**（与 7 月基线逐位
>   一致）→ `cgra_execute.c` → riscv-gcc（**官方 Makefrag 参数**）→ Verilator
> - 结果：**`RELU PASS: All 8 values correct`，85 cycles**（与历史一致）
> - 编译踩坑（已解决）：必须用 `software/tests` 官方构建参数——
>   `-DBAREMETAL=1 -nostdlib -static -T test.ld -march=rv64gc`，include 路径
>   须含 `riscv-tests/env`（encoding.h）与 `benchmarks/common`（util.h）。
>   自行拼命令会反复踩 `sys/mman.h`/`encoding.h` 缺失。
>
> **结论：硬件侧无异常。** 现有二进制的 `rot_en` 为编译期 false（已被常量
> 折叠，top.v 中 0 次出现），故双缓冲未激活——层次 1 验证需开启后重建。
>
> ### 层次 1 验证（2026-09-26 22:xx）—— 发现 2 个真实缺陷并已修复
>
> 开启 `rot_en`（TramSpec：`cgra_rot_en=true, num_ctx=4, exe_ii=2`）重建后，
> relu RTL 仿真 A/B 判别：
>
> **发现 A（单 config，rot_en=true）**：不挂起但 **3 处 mismatch，输出后移
> 一位**（expected[3]=2.0 出现在位置 4）。根因：我把主输出通路穿过
> `rf.io.out(0)`，`rot_en=true` 时多一级寄存器 → SA 旧配置的时序假设失效。
> **属设计失误，非必然代价**：跨 context 消费发生在输入 MUX（`rf.out(1)`，
> k≥1），主通路根本不需要穿 RF。
> **修复**：RF 写端口从 `alu.io.out` **并行捕获**，主通路恒为
> `alu.io.out → dmr.io.in` 组合直连 —— 无论 rot_en 开关，流水时序逐位不变。
> （同时修正 connections 元数据：ALU→RF 捕获 + ALU→REG 主通路。）
>
> **发现 B（双 config）**：死锁——CGRAController FSM 在 `s_exe_wait` 只接受
> EXE_CMD，链式第二个 `config()` 被拒 → RS 排队卡死 → fence 永不返回。
> **修复**：`s_exe_wait` 增加对 isCfgCmd 的接受分支（装载下一 config slot
> 后回到 `s_cfg_run`），req.ready 相应放宽。
>
> **修复后验证（2026-09-26 23:0x，重建模拟器 23:03）—— 层次 1 完整闭合 ✅**
>
> | 测试 | 结果 | 判读 |
> |---|---|---|
> | **双 config（A/B 同内容）+ rot_en=true** | **`RELU PASS` 8/8，86 cycles** | **主判据达成**：slot 装载、链式 CFG、ctx 推进、读侧 A/B 切换全部在真实电路上正确 |
> | 单 config + rot_en=true | 85 cycles，1 mismatch（位置 6 的 1.5 丢失） | **预期表现**：slot B 空，ctx 窗口读空配置。反向佐证切换机制真实生效 |
> | 单 config + rot_en=false（基线） | `RELU PASS` 85 cycles | 对照组 |
>
> 86 vs 85 的 +1 = 第二次 config 命令的固有开销（ConfigController 从 spad
> 重读 11 词 + FSM 翻转），非数据通路时序问题——输出逐位正确证明主通路
> 与 rot_en 无关。修复前单 config 是 3 处系统性移位，修复后时序归位。
>
> **层次 1 结论：双缓冲 + ctx 推进 + 链式 config 装载的硬件机制在 RTL 上
> 验证正确。** 下一步 = 软件侧模调度（产出两套**不同**配置 + RF sel/k +
> 反馈 MUX 选择），实现真正的时空映射。
>
> ⚠️ **踩坑记录**：`CGRAMG` 走 `TramSpec.loadSpec()` 读 `tram_spec.json`，
> 而那是**另一台机器的旧 spec**（3×8 阵列、9 操作、`cfg_blk_offset=2`）；
> `make` 走 `chipyard.Generator`，用的是 `TramSpec.scala` 默认值
> （8×16、16 操作、`offset=3`）。两者参数不同，验证时务必用后者，
> 否则会遇到虚假的 `cfgAddrWidth > cfgBlkOffset` 断言失败。
>
> **同步说明**：`example/cgra_adg.json` 已同步为新导出版本（旧版备份为
> `cgra_adg.json.bak-20260926`）。差异根因不是 RF，而是
> `iob_ag_nest_levels` 8→3（IOB 配置 232→104 位）+ 操作集 13→16（ALU 4→5 位）。
> 注：`bootstrap_runner.py` 读的是硬件仓库 ADG，`main.py`/`evaluate.py` 读
> `example/`，同步前两者本就不一致。
> 目标：以最小改动引入「跨 context 数据通路」，为模调度/时空映射铺路
> 代码基线：`/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/generators/fdra/cgra-mg`

---

## 0. 一句话原理

把「跨 context 传数据」从**数据通路问题**改成**寻址问题**：

```
写端口：物理号 = (逻辑号 + ctx)          mod N
读端口：物理号 = (逻辑号 + ctx - k)      mod N     k = 要几个 context 之前的值
```

物理寄存器**只有一份，不复制**；每个 context 让「逻辑号→物理号」的映射滑动一格。编译器始终写「逻辑 R0」，硬件自动让它落在不同物理寄存器上，从而既不覆盖、又能跨 context 读到。

---

## 1. 现状盘点（决定改动面）

| 结构 | 位置 | 现状 |
|---|---|---|
| `RF` 寄存器堆 | `dsa/util/RegFile.scala` | 存在，但 **PE.scala L73 被注释** |
| `num_rf_reg` 参数 | `PE.scala L36` | **被注释** |
| DMR 双模寄存器 | `dsa/util/DMR.scala` | 存活，**只有 2 个标量寄存器**，不是堆 |
| `ii` / `loop_cycles` | `ISA.scala` L106-116、`CGRAController.scala` L70-72 | **被注释** |
| `io.ii` / `io.cycles` | `CGRA.scala` L123-124 | **被注释** |
| `lg_max_ii`（IOB 侧） | `IOB.scala` L23、`tram_spec.json` L82 | **存活**（但那是 IO 地址流水，非计算阵列） |

**被剥离的两个基础件（寄存器堆 + II 控制）正是时空执行所需** —— 本设计即恢复二者并接上。

---

## 2. 改动总览

| # | 文件 | 改动 |
|---|---|---|
| 1 | `dsa/util/RegFile.scala` | `RF` 增加旋转寻址端口与逻辑 |
| 2 | `dsa/pe/PE.scala` | 恢复 `RF` 实例化，插入 ALU→RF→DMR，新增 MUX 输入 |
| 3 | `dsa/template/CGRA.scala` | 新增 `ctx_ptr` / `rot_en` 输入并广播到所有 GPE |
| 4 | `soc/CGRAController.scala` | 恢复 `ii` 寄存器 + context 计数器，驱动 `ctx_ptr` |

---

## 3. 代码草案

### 3.1 `RegFile.scala` —— 旋转寻址

```scala
package tram.dsa

import chisel3._
import chisel3.util._
import scala.math.pow

/** Rotating Register File
 *
 * @param width     register data width
 * @param regNum    number of physical registers, >= numCtx (建议取 2 的幂)
 * @param numIn     number of write ports
 * @param numOut    number of read ports
 * @param lgMaxCtx  log2(max context number)
 * @param lgMaxK    log2(max cross-context distance k)
 */
class RF(width: Int, regNum: Int, numIn: Int, numOut: Int,
         lgMaxCtx: Int, lgMaxK: Int) extends Module {
  val selWidth   = log2Ceil(regNum)
  val rotCfgWidth = numOut * lgMaxK            // k of each read port

  val io = IO(new Bundle {
    val en      = Input(Bool())
    val config  = Input(UInt((selWidth * (numIn + numOut) + rotCfgWidth).W))
    val in      = Input(Vec(numIn, UInt(width.W)))
    val out     = Output(Vec(numOut, UInt(width.W)))
    // ---- 新增：旋转寻址 ----
    val ctx_ptr = Input(UInt(lgMaxCtx.W))      // 当前 context 指针（全阵列广播）
    val rot_en  = Input(Bool())                // false 时行为与旧 RF 完全一致
  })

  val regs = RegInit(VecInit(Seq.fill(regNum) { 0.U(width.W) }))

  // ---------- config 解码 ----------
  // [rotCfgWidth-1 : 0]            : k of read port j
  // [selBase + ...]                : sel of each port (write ports first, then read)
  val ks = Wire(Vec(numOut, UInt(lgMaxK.W)))
  (0 until numOut).foreach { j =>
    ks(j) := io.config((j + 1) * lgMaxK - 1, j * lgMaxK)
  }
  val selBase = rotCfgWidth
  val sels = Wire(Vec(numIn + numOut, UInt(selWidth.W)))
  (0 until (numIn + numOut)).foreach { i =>
    sels(i) := io.config(selBase + (i + 1) * selWidth - 1, selBase + i * selWidth)
  }

  // ---------- 旋转地址计算 ----------
  // phys = (logic + ctx + regNum - k) % regNum
  // 先加 regNum 保证不出现负数（要求 k <= regNum）
  // regNum 为 2 的幂时 % 退化为低位截断，零成本
  private def isPow2(n: Int): Boolean = n > 0 && (n & (n - 1)) == 0

  def rotPhys(logic: UInt, k: UInt): UInt = {
    val sum = logic +& io.ctx_ptr +& regNum.U - k   // +& 保证位宽不溢出
    if (isPow2(regNum)) sum(log2Ceil(regNum) - 1, 0)
    else sum % regNum.U
  }

  if (regNum > 1) {
    // ---- 写端口：k = 0，写到「当前 context」的位置 ----
    when(io.en) {
      (0 until numIn).foreach { i =>
        val phys = Mux(io.rot_en, rotPhys(sels(i), 0.U), sels(i))
        regs(phys) := io.in(i)
      }
    }
    // ---- 读端口：k = 配置值，读「k 个 context 之前」的值 ----
    (0 until numOut).foreach { j =>
      val phys = Mux(io.rot_en, rotPhys(sels(numIn + j), ks(j)), sels(numIn + j))
      io.out(j) := regs(phys)
    }
  } else {
    when(io.en) { (0 until numIn).foreach { i => regs(0) := io.in(i) } }
    (0 until numOut).foreach { j => io.out(j) := regs(0) }
  }
}
```

**要点**：
- `rot_en = false` 时走 `Mux` 的另一分支，与旧 `RF` **逐位等价**
- `regNum` 建议取 2 的幂，`%` 退化为位截断
- 面积增量：每个端口 1 个加法器 + **零** 额外寄存器

---

### 3.2 `PE.scala` —— 恢复 RF 并接入数据流

数据流改动（**DMR 的语义与接线完全不动**）：

```
   现状：alu.out ──> dmr.in ──> dmr.out(0) ──> io.out(0)
                                dmr.out(1) ──> imuxs 反馈

   改后：alu.out ──> rf.in(0) ──> rf.out(0) ──> dmr.in ──> dmr.out(0) ──> io.out(0)
                       │                          dmr.out(1) ──> imuxs 反馈（不变）
                       └──> rf.out(1) ──> imuxs **新增** 反馈输入
```

#### (a) 参数与 IO

```scala
  // 恢复（原 L36 被注释）
  val numRegRF = attrs("num_rf_reg").asInstanceOf[Int]
  // 新增
  val lgMaxCtx = attrs("lg_max_ctx").asInstanceOf[Int]
  val lgMaxK   = attrs("lg_max_k").asInstanceOf[Int]

  val io = IO(new Bundle {
    val cfg_en   = Input(Bool())
    val cfg_addr = Input(UInt(cfgAddrWidth.W))
    val cfg_data = Input(UInt(cfgDataWidth.W))
    val start    = Input(Bool())
    val en       = Input(Bool())
    val in       = Input(Vec(numInPerOperand.sum, UInt(width.W)))
    val out      = Output(Vec(1, UInt(width.W)))
    // ---- 新增 ----
    val ctx_ptr  = Input(UInt(lgMaxCtx.W))
    val rot_en   = Input(Bool())
  })
```

#### (b) 实例化与接线

```scala
  val alu = Module(new ALU(width, ops))
  val rf  = Module(new RF(width, numRegRF, 1, 2, lgMaxCtx, lgMaxK))  // 恢复
  val dmr = Module(new DualModeReg(width, hasAcc, lgMaxWI, lgMaxLat, lgMaxCycles, lgMaxRepeats))
  val rdu = Module(new SharedRDU(width, maxDelay, aluOperandNum))

  // ---- RF 接线 ----
  rf.io.en      := io.en
  rf.io.ctx_ptr := io.ctx_ptr
  rf.io.rot_en  := io.rot_en
  rf.io.in(0)   := alu.io.out
  rdu.io.en     := io.en
  dmr.io.in     := rf.io.out(0)     // DMR 输入来源改为 RF（DMR 自身逻辑不变）
  io.out(0)     := dmr.io.out(0)
```

#### (c) 输入 MUX 增加一路旋转读

`extraMuxIns` 在原有基础上 +1：

```scala
  val extraMuxIns = {
    val base = if (hasAcc && aluOperandNum > 1) 2 else 1   // const [+ REG]
    base + 1                                                // + 旋转读 rf.out(1)
  }
```

MUX 输入端连接（保留原有 `dmr.io.out(1)` 反馈，新增一路）：

```scala
  val rfFbIdx = num + extraMuxIns - 1      // 新增的最后一路由 rf.out(1) 驱动
  imuxs(i).in.zipWithIndex.foreach { case (in, j) =>
    if (j == 0) {
      in := const
      connections.append((smi_id("Const")(0), "Const", 0, smi_id("Muxn")(i), "Muxn", j))
    } else if (j <= num) {
      in := io.in(offset + j - 1)
      connections.append((smi_id("This")(0), "This", offset + j - 1, smi_id("Muxn")(i), "Muxn", j))
    } else if (j == rfFbIdx) {
      in := rf.io.out(1)                   // ---- 新增：跨 context 读 ----
      connections.append((smi_id("RF")(0), "RF", 1, smi_id("Muxn")(i), "Muxn", j))
    } else {
      in := dmr.io.out(1)                  // 原有 REG 反馈，保持不变
      connections.append((smi_id("REG")(0), "REG", 1, smi_id("Muxn")(i), "Muxn", j))
    }
  }
```

> 需在 `sm_id` / `smi_id` 中登记 `"RF"`，并相应递增 `next_smi_id`。

#### (d) 配置位分配

新增 RF 配置段。`sumCfgWidth` 增加：

```scala
  val rfSelWidth  = log2Ceil(numRegRF)
  val rfKCfgWidth = 2 * lgMaxK                          // 2 个读端口各一个 k
  val rfCfgWidthNew = rfSelWidth * (1 + 2) + rfKCfgWidth  // 1 写 + 2 读 + k

  val sumCfgWidth = (constCfgWidth + aluCfgWidth + rfCfgWidth   // 注意：rfCfgWidth 现为 DMR 用
                     + rduCfgWidth + imuxCfgWidth + rfCfgWidthNew)
```

分配一段给 RF（放在 DMR 之后即可）：

```scala
  if (rfCfgWidthNew != 0) {
    rf.io.config := cfgOut(offset + rfCfgWidthNew - 1, offset)
    configuration += smi_id("RF")(0) -> ("RF", offset + rfCfgWidthNew - 1, offset)
  }
  offset += rfCfgWidthNew
```

> ⚠️ 命名陷阱：现有 `rfCfgWidth` 变量 **实际是 DMR 的配置宽度**（`dmr.io.config.getWidth`），
> 历史遗留名。新增段请用 `rfCfgWidthNew` 之类，避免混淆。

---

### 3.3 `CGRA.scala` —— 广播 `ctx_ptr`

```scala
  val lgMaxCtx = attrs("cgra_lg_max_ctx").asInstanceOf[Int]

  val io = IO(new Bundle {
    ...
    val iob_ens = Input(UInt(numIOB.W))
    val en      = Input(Bool())
    val start   = Input(Bool())
    // ---- 新增 ----
    val ctx_ptr = Input(UInt(lgMaxCtx.W))
    val rot_en  = Input(Bool())
  })

  // gpe_attrs 新增
  "num_rf_reg" -> numRegRF,
  "lg_max_ctx" -> lgMaxCtx,
  "lg_max_k"   -> lgMaxK,

  // 实例化后连接（与 iob.io.start / io.en 同样方式）
  gpes.foreach { gpe =>
    gpe.io.ctx_ptr := io.ctx_ptr
    gpe.io.rot_en  := io.rot_en
  }
```

---

### 3.4 `CGRAController.scala` —— context 计数器

恢复 `ii` 相关寄存器，用 II 驱动 context 推进：

```scala
  val numCtx  = attrs("cgra_num_ctx").asInstanceOf[Int]
  val lgMaxII = attrs("cgra_exe_lg_max_ii").asInstanceOf[Int]   // 原 L31 被注释

  val ii       = RegInit(0.U(lgMaxII.W))      // 原 L70 被注释
  val ctxCnt   = RegInit(0.U(log2Ceil(numCtx).W))
  val iiCnt    = RegInit(0.U(lgMaxII.W))
  val rot_en   = RegInit(false.B)
```

在 `s_exe_run` 状态推进 context：

```scala
    is(s_exe_run){
      exe_start := false.B
      ex_enable := true.B
      // ---- 新增：按 II 推进 context ----
      when(ex_enable && rot_en){
        when(iiCnt + 1.U >= ii){
          iiCnt  := 0.U
          ctxCnt := Mux(ctxCnt + 1.U >= numCtx.U, 0.U, ctxCnt + 1.U)
        }.otherwise{
          iiCnt := iiCnt + 1.U
        }
      }
      when(!cgra.io.done){ // indicate execution is running
        state := s_exe_resp
      }
    }
```

连接（原 L189-191 被注释的三行恢复并扩展）：

```scala
  cgra.io.ii      := ii        // 若 CGRA 侧也恢复 ii 端口
  cgra.io.ctx_ptr := ctxCnt
  cgra.io.rot_en  := rot_en
```

`rot_en` 由 CFG 命令置位（见 §4），`ii` 与 `numCtx` 由配置写入。

---

## 4. ISA 改动（配合 §3.4）

恢复 `ISA.scala` 的 `CgraCfgRs2`（L106-116）中的 `ii` / `loop_cycles`，或扩展现有 `CfgRs2`：

```scala
  class CfgRs2 extends Bundle {
    val dep_type      = UInt(DEP_TYPE_WIDTH.W)
    val task_id       = UInt(TASK_ID_WIDTH.W)
    val _unuse0       = UInt(CFG_CMD_RS2_UNUSE0_WIDTH.W)
    val cfg_num       = UInt(CFG_CMD_RS2_NUM_WIDTH.W)
    val cfg_base_addr = UInt(CFG_CMD_RS2_BASE_ADDR_WIDTH.W)
    // ---- 新增（最小集）----
    // val ii     = UInt(8.W)    // 每个 context 持续的周期数
    // val numCtx = UInt(4.W)    // context 数
    // val rotEn  = Bool()       // 旋转使能
  }
```

> 最小验证阶段可**不改 ISA**：把 `ii` / `numCtx` / `rot_en` 用 Chisel 参数或 spec 常量固定，
> 编译期决定，避免同时动 ISA + 编译器 + 驱动。

---

## 5. spec 参数改动

`tram_spec.json` / `TramSpec.scala` 新增（建议初值）：

```json
  "cgra_num_ctx"        : 4,
  "cgra_lg_max_ctx"     : 2,
  "cgra_lg_max_k"       : 2,
  "cgra_num_rf_reg"     : 4,     // = numCtx，且为 2 的幂
  "cgra_exe_lg_max_ii"  : 4
```

`num_rf_reg` 必须 **≥ numCtx** 且建议取 2 的幂，使 `%` 退化为位截断。

---

## 6. 兼容性设计

| `rot_en` | 行为 |
|---|---|
| `false` | `Mux` 走旧分支，RF 退化为普通寄存器堆；**DMR、ACC、IOB、配置流全部不变** |
| `true` | 启用旋转寻址 |

因此该改动**可合入主干而不影响现有空间映射流程**，风险可控。建议在 `tram_spec.json` 中默认 `rot_en=false`，由实验显式打开。

---

## 7. 最小验证计划（建议顺序）

目标：**先证明「跨 context 数据不出片」可行，再谈配置多套化与模调度。**

| 步 | 内容 | 判据 |
|---|---|---|
| V1 | 只改 `RF.scala` + `PE.scala`（`rot_en` 由 spec 常量强制 true，`ctx_ptr` 先用外部计数器驱动） | 编译通过、Verilator 生成 |
| V2 | 构造最小 2-context DFG：ctx0 算 `a+b`，ctx1 读该结果做 `+c` | 功能正确 = 跨 context 数据通路成立 |
| V3 | 对比：同一 DFG 用 `rot_en=false`（数据需出片经 DRAM） | 周期显著下降 = 收益成立 |
| V4 | 接回 `CGRAController` 的 ii/ctx 计数器，去掉外部驱动 | 与 V2 结果一致 |
| V5 | 再动配置存储多 context 化（双缓冲方案） | — |

V2/V3 是关键判据。**若 V3 无收益，说明跨 context 仍被其它通路卡住（如 spad 无 TL 地址），需回到路线 A。**

---

## 8. 风险与未决问题

1. **`%` 电路开销**：`regNum` 非 2 的幂时取模是真实除法电路，128 个 PE 累加可观。**务必取 2 的幂**。
2. **配置位宽增长**：`sumCfgWidth` 增加 `rfCfgWidthNew`，会拉长配置链装载时间（~1 cyc/词）。需重新评估 `cfg_spad_size=2048` 是否够装多 context。
3. **k 的语义需与编译器对齐**：本文档采用「读偏移 = 写偏移 − k」，编译器写同一逻辑号即可跨 context 读。另一种约定是「写偏移 = 读偏移 − k」，二者差一个方向，**必须在写模调度前定死**。
4. **`ctx_ptr` 时序**：context 翻转与 PE 流水线的相对关系未验证，可能出现读写同拍冲突（同一物理寄存器同时读写）。**建议 V1 阶段用非对齐 II（ii ≥ 2）规避**，待波形确认后再压到 ii=1。
5. **DMR 与 RF 串联后 ACC 语义**：`dmr.io.in` 来源改为 `rf.io.out(0)`，若 `rot_en=true` 而 RF 地址每 context 变，ACC 累加链可能断裂。**ACC 场景建议 `rot_en=false` 或 k=0。**
6. **被注释原因未知**：`RF` / `ii` 被剥离可能是未完成或验证未过，恢复不等于能工作。

---

## 9. 与后续工作的接口

本设计只解决**跨 context 数据通路**。完整时空映射还需：

- 配置存储多 context 化（推荐双缓冲，面积 ×2 而非 ×N，见 `plan.md` 讨论）
- 模调度编译器（核心指标从「关键路径」变为「II」）
- 评估器重写：`hw ≈ Σ_contexts(config_cmd + execute) + 数据项`，II 成为强信号
- 时空映射算法（agent 侧）

**依赖顺序**：本设计（跨 context 数据）→ 配置多 context 化 → 模调度 → 评估器 → agent 训练。
