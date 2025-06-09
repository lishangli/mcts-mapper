# 📦 MCTS-MAP
A CGRA Mapping Algorithm Project Based on MCTS-RL

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
│   ├── trainner#	# train class
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
