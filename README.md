# 📦 MCTS-MAP
A CGRA Mapping Algorithm Project Based on MCTS-RL

## 📈 Results on the RTL simulator (2026-11)

| kernel | verification | cost | throughput |
|---|---|---|---|
| conv2d_3x3 (real convolution, AG streamed) | **42/42 outputs bit-exact** | 178–195 cycles | **18.1–19.8 ops/cycle (28–31 % of the 64 ops/cycle array peak)** |
| nn4x2 (4 inputs → 2 outputs, stream sharing on) | `[18, 21]` bit-exact | 653 / 915 / 1406 cycles @ 64 / 128 / 256 samples | 1.57 / 2.24 / **2.91** |
| nn3x2 | `[10, 14]` bit-exact | 530 cycles @ 64 samples | 1.45 |

(A 20-node DAG used to be listed at 698 cycles / 3.67 ops/cycle. It no longer
maps — 32 minutes with a 1800 s budget, sharing on and off — so that row is
withdrawn rather than quoted.)

### Two ceilings, and why they differ

```
dense flow (nn4x2, stream sharing on):
  64 / 128 / 256 samples -> 1.57 / 2.24 / 2.91 ops/cycle
  marginal 128->256      -> 3.84 cycles/sample, of which 2.85 is store
  -> ~1.43 cycles per OUTPUT word, ~0.21 per input word, exec = 6 cycles
  -> result-drain bound: 8 ops/output word / 1.43 = 5.6 marginal, 2.91 average
  (the older ~6 figure came from the retired 20-node DAG)

conv2d (operands from the AG window):
  18 ops/output word / 1.0 cycle per output word = 18.1 ops/cycle
  3528 ops in 195 cycles, reproducible with `tools/conv/run.sh`
```

So both kernels fit one parameter, **ops per output word divided by the cycles an
output word costs**: dense 8 / 1.43, conv 18 / 1.0. Raising the dense ratio needs
`2 x inputs` nodes per output, which the mapper cannot place (20–24 node ceiling),
and the only other lever — computing several outputs per operand word — needs an
II schedule. `S2_PACK=1`, the crude version of it, fails to map (`no const IOB`),
because folding is incompatible with the context/phase constraint. See
`docs/AG_FED_DENSE.md`.

### Known limitations (each with its evidence)

| limitation | evidence |
|---|---|
| one-sample operand skew on **varying** data | `dfg_quad` ramp fits `capture[i] = ref(i-L)+s` with s = -1; constant stimuli cannot see it, which is why every regression passes |
| a rare placement emits a program that does not terminate | 1 in ~6 `dfg_quad` runs; bounded by `S2_SIM_TIMEOUT`/`S2_SIM_RETRY`/`S2_CASE_TIMEOUT`, and orphaned children are killed with `killpg` |
| alignment fixes the skew but is unusable at layer scale | `dfg_quad` skew 0 (rmse 0.00), but `nn3x2` fails on every attempt; parked, see `docs/STATUS.md` |
| mapping capacity: 20–24 nodes is the practical ceiling | `nn4x2` (32 nodes) maps in 653–2400 s depending on the draw; `l4w5o3c3` (20 nodes) no longer maps at all; larger DFGs never emit |
| readback: ~1 KiB per launch | two capture channels of 1024 B never emit, two of 512 B or one of 1024 B do (`S2_STORE_BYTES` is now clamped automatically) |

### Next real step

The performance wall is `2 x inputs` ops per output word at `1.43` cycles per word,
and both escapes (more nodes per output, several outputs per operand word) are
blocked by the mapper: the first by capacity, the second by the context/phase
constraint that `S2_PACK` hits. Pick up at **a phase-aware joint scheduler**:
assign each node's context together with its operand routes instead of
re-routing afterwards, so one operand word can feed several outputs within II.
`docs/ITERATION_PLAN.md` has the shape of it, `docs/AG_FED_DENSE.md` the numbers.

Models, experiments and negative results are in `docs/STATUS.md`,
`docs/EXPERIMENTS.md`, `docs/EXECUTION_MODEL.md`, `docs/NN_WORKLOADS.md`,
`docs/AG_FED_DENSE.md` (why dense stops at ~3 ops/cycle), `docs/MRRG_PLAN.md`
and `docs/RELATED_WORK.md`.

## 🚀 Features
✨ Feature 1: Supports front-end compiler CGRA-Compiler

⚙️ Feature 2: Utilizes MCTS and various RL algorithms

🔍 Feature 3: Supports conversion of mapping results to actual configuration code (to be refined)

## 🔧ToDO

- [ ] Fix some real hardware configuration errors 
- [ ] Add stronger strategy for routing and placement

## 🛠️ Installation

### Install CGRA-Mapper backend
🚀Install back-end package 'cgra_compiler-0.1.1-cp38-cp38-linux_x86_64.whl'

### Dependencies Install
```bash
pip install -r requirements.txt
```

## Project Structure
```text
.
├── dataset 		# Dataset 
│   ├── augment
│   ├── collect_data
│   └── microbench
│       ├── accumulate
│       ├── cap
│       ├── conv2
│       ├── conv3
│       ├── json
│       ├── mac
│       ├── mac2
│       ├── matrixmultiply
│       ├── mults1
│       ├── mults2
│       ├── nomem1
│       ├── nomem2
│       ├── simple
│       ├── simple2
│       ├── sum
│       └── two_loops
├── example 		# Some toy examles
├── exp_data
├── figures			# training figures
│   └── data
├── models
├── src 			# core code
│   ├── agent		# MCTS agent
│   ├── env			# environment
│   ├── mapper		# CGRA mapper
│   ├── models		# Network models
│   ├── parser		# JSON Parser
│   ├── trainer	# train class
│   └── utils
└── videos
```


## Usage

### CGRA Mapping Usage
```python
from adgParser import *
from dfgParser import *
from mapping import Mapping
from operations import Operations

dfg_parser = DFGParser("dfg.json")
adg_parser = ADGIR("cgra_adg.json")
operations = Operations()
operations.OpParser("operations.json")
dfg = dfg_parser.getDFG()
adg = adg_parser.getADG()

model_path = "models/best-agent.pt"

map = Mapping(dfg, adg, operations, model_path)

map.run(config = "export_config_path")
```

### CGRA configuration C++ code generation

C++ code genenration usage example is below:
```python
from cgra_compiler_python import *

ops = Operations.Instance("operations.json")
air = ADGIR("cgra_adg.json")
dir = DFGIR("dfg.json")

adg = air.getADG()
dfg = dir.getDFG()

map = MapperSA(adg, dfg, 0, 0, True)
config_dir = "."
config_func = "cgra_execute"
config_export_file = "config.json"

map.executeBind(True, False, ".", "cgra_execute", "config.json")
```
C++ code example for CGRA configuration is below:
```c++
void cgra_execute(void** din_addr, void** dout_addr)
{
	 #ifdef CGRA_DEBUG
	 printf("|");
	 #endif
	static unsigned short cin[45][3] __attribute__((aligned(16))) = {
		{0x6000, 0x0002, 0x0010},
		{0x0080, 0x0000, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x0000, 0x0013},
		{0x0000, 0x0000, 0x0014},
		{0x0000, 0x0000, 0x0015},
		{0x0000, 0x0200, 0x0016},
		{0x0000, 0x0000, 0x0017},
		{0x8000, 0x0002, 0x0028},
		{0x0080, 0x0000, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0000, 0x002b},
		{0x0000, 0x0000, 0x002c},
		{0x0000, 0x0000, 0x002d},
		{0x0000, 0x2e00, 0x002e},
		{0x0041, 0x0000, 0x002f},
		{0x4000, 0x0002, 0x0038},
		{0x0080, 0x0000, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0000, 0x003b},
		{0x0000, 0x0000, 0x003c},
		{0x0000, 0x0000, 0x003d},
		{0x0000, 0x0200, 0x003e},
		{0x0000, 0x0000, 0x003f},
		{0x0000, 0x0002, 0x0040},
		{0x0080, 0x0000, 0x0041},
		{0x0000, 0x0000, 0x0042},
		{0x0000, 0x0000, 0x0043},
		{0x0000, 0x0000, 0x0044},
		{0x0000, 0x0000, 0x0045},
		{0x0000, 0x0200, 0x0046},
		{0x0000, 0x0000, 0x0047},
		{0x2000, 0x0002, 0x0048},
		{0x0080, 0x0000, 0x0049},
		{0x0000, 0x0000, 0x004a},
		{0x0000, 0x0000, 0x004b},
		{0x0000, 0x0000, 0x004c},
		{0x0000, 0x0000, 0x004d},
		{0x0000, 0x0200, 0x004e},
		{0x0000, 0x0000, 0x004f},
		{0x2028, 0x0000, 0x0121},
		{0x0001, 0x0000, 0x0148},
		{0x000a, 0x0001, 0x0149},
		{0x1008, 0x0000, 0x0151},
		{0x1008, 0x0000, 0x0169},
	};

	load_cfg((void*)cin, 0x100000, 270, 0, 0);
	 int sizes_2[1] = {512};
	 int strides_2[1] = {1};
	load_mem(din_addr[0] + 0, 0x0, 0, 0, 0, strides_2, sizes_2, 1, 0);
	 int sizes_3[1] = {512};
	 int strides_3[1] = {1};
	load_mem(din_addr[1] + 0, 0x8000, 0, 0, 0, strides_3, sizes_3, 1, 0);
	 int sizes_4[1] = {512};
	 int strides_4[1] = {1};
	load_mem(din_addr[2] + 0, 0x10000, 0, 0, 0, strides_4, sizes_4, 1, 0);
	 int sizes_5[1] = {512};
	 int strides_5[1] = {1};
	load_mem(din_addr[3] + 0, 0x18000, 0, 0, 0, strides_5, sizes_5, 1, 0);
	config(0x0, 45, 0, 0);
	execute(0x1d2, 0, 0);
	 int sizes_10[1] = {512};
	 int strides_10[1] = {1};
	store_mem(dout_addr[0] + 0, 0x20000, 0, 0, strides_10, sizes_10, 1, 0);
fence(1);}
```

This file can used to map the dfg in C/C++/Python application to CGRA Core in the [CGRA-Compiler Project](https://github.com/lishangli/wafer-compiler-for-cgra)

## Lincense
```text
MIT License

Copyright (c) [2025] [Shangli Li]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
