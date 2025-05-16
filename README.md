# 📦 HMapZero

一个基于MCTS-RL的CGRA映射算法项目

## 🚀 Features

- ✨ Feature1, 支持前端编译器CGRA-Compiler
- ⚙️ 特性2，使用MCTS和不同RL算法
- 🔍 特性3，支持将映射结果转换为实际配置代码

## 🛠️ Installation

### 使用 CGRA-Mapper backend
🚀安装后端cgra_compiler-0.1.1-cp38-cp38-linux_x86_64.whl

### 安装依赖
```bash
pip install -r requirements.txt
```

## 项目目录
```text
.
├── GAT.py
├── README.md
├── __pycache__
├── adg.py
├── adgParser.py
├── cgra_adg.json
├── cgra_call.txt
├── cgra_compiler-0.1.0-cp38-cp38-linux_x86_64.whl
├── cgra_compiler-0.1.1-cp38-cp38-linux_x86_64.whl
├── cgra_compiler.whl
├── cgra_execute.c
├── config.json
├── config_gen.py
├── dataLoader.py
├── dataset
├── dfg.json
├── dfg.py
├── dfgParser.py
├── dfg_simple.json
├── env.py
├── envi.py
├── exp_data
├── figures
├── libopenh264-2.5.0-linux64.7.so.bz2
├── loadData.py
├── load_data.py
├── mapping.py
├── mcts.py
├── model.py
├── model_structure
├── model_structure.png
├── models
├── net.py
├── nohup.out
├── operations.json
├── operations.py
├── ppo.py
├── profile_output.lprof
├── profile_output.txt
├── profile_output_2024-12-17T164817.txt
├── profile_output_2024-12-17T165234.txt
├── profile_output_2024-12-17T180148.txt
├── profile_output_2024-12-17T181118.txt
├── profile_output_2024-12-17T183151.txt
├── profile_output_2024-12-17T184059.txt
├── profile_output_2024-12-17T184321.txt
├── profile_output_2024-12-17T184607.txt
├── profile_output_2024-12-17T191234.txt
├── profile_output_2024-12-17T191506.txt
├── profile_output_2024-12-17T191858.txt
├── pyproject.toml
├── state.py
├── test.py
├── test.py.lprof
├── train.log
├── train.py
├── ttest.py
├── utils.py
├── videos
└── visual.py
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
