#!/bin/bash
# build + run one DFG through the MCTS bridge -> S2 emission -> RTL.
# usage: ./run_dfg.sh <dfg.json> <tag> [--ramp]
#
# Environment (all optional; defaults match the reference machine):
#   FDRA             root of the FDRA generator tree (holds software/tests)
#   FDRA_SIM         verilator simulator directory
#   RISCV_TOOLCHAIN  riscv64-unknown-elf-gcc prefix
#   CONDA_ENV_BIN    python env holding the cgra_compiler wheel deps
#   S2_OUT_DIR       where artefacts are written (default: this directory)
set -e
DFG=${1:?usage: run_dfg.sh <dfg.json> <tag> [--ramp]}
TAG=${2:?usage: run_dfg.sh <dfg.json> <tag> [--ramp]}
S2=$(cd "$(dirname "$0")" && pwd)
export S2_OUT_DIR=${S2_OUT_DIR:-$S2}
FDRA=${FDRA:-/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/generators/fdra}
RISCV=${RISCV_TOOLCHAIN:-/home/lsl/wafer-compiler-for-cgra/simulator/riscv-tools/riscv64-unknown-elf-gcc-8.3.0-2020.04.1-x86_64-linux-ubuntu14}
T=$FDRA/software/tests
SIM=${FDRA_SIM:-/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/sims/verilator}
export PATH=$RISCV/bin:${CONDA_ENV_BIN:-/home/lsl/miniconda3/envs/cgra38/bin}:$PATH

DFG=$(readlink -f "$DFG")   # the bridge chdir()s to mcts-mapper/src: keep DFG absolute
cd "$S2"
python mcts_bridge.py "$DFG" 2>&1 | tee "${TAG}_bridge.log"
if [ "${3:-}" = "--ramp" ]; then
  sed -i 's/for (int i = 0; i < 64; i++) c0\[i\] = .*/for (int i = 0; i < 64; i++) c0[i] = i;/' "$S2_OUT_DIR/main_s2.c"
fi

riscv64-unknown-elf-gcc -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany -std=gnu99 -O2 \
  -ffast-math -fno-common -fno-builtin-printf -march=rv64gc -Wa,-march=rv64gcxhwacha \
  -I$S2 -I$T/riscv-tests -I$T/riscv-tests/env -I$T -I$T/include -I$T/riscv-tests/benchmarks/common \
  -nostdlib -nostartfiles -static -T $T/riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1 \
  "$S2_OUT_DIR/main_s2.c" "$S2_OUT_DIR/cgra_execute.c" \
  $T/riscv-tests/benchmarks/common/*.c $T/riscv-tests/benchmarks/common/*.S \
  -lm -lgcc -o "$S2_OUT_DIR/$TAG.riscv"

(cd "$SIM" && timeout 900 ./simulator-chipyard-TramSoCConfig "$S2_OUT_DIR/$TAG.riscv" > "$S2_OUT_DIR/${TAG}_run.log" 2>&1) || true
python3 - "$S2_OUT_DIR/${TAG}_run.log" <<'PY'
import re, sys
t = open(sys.argv[1]).read()
v = [int(x) for x in re.findall(r"O\d*\[\d+\] = (-?\d+)", t)]
print("O[8:40] =", v[8:40])
print("steady unique =", sorted(set(v[20:120])))
for line in t.splitlines():
    if line.startswith("S2_PERF"):
        print(line)
PY
