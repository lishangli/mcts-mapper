#!/bin/bash
# Build, run and verify the conv2d_3x3 kernel on the RTL simulator.
#
#   ./run.sh                 # build from the FDRA tree, run, check, print throughput
#   FDRA=... RISCV_TOOLCHAIN=... ./run.sh
#
# Reports the cycle count around cgra_execute0() and checks all defined outputs
# bit-exactly against the C reference (column-major tap order, see
# tools/s2/check_conv2d.py). 3528 ops / 178 cycles = 19.8 ops/cycle on the
# reference machine.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
FDRA=${FDRA:-/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/generators/fdra}
RISCV=${RISCV_TOOLCHAIN:-/home/lsl/wafer-compiler-for-cgra/simulator/riscv-tools/riscv64-unknown-elf-gcc-8.3.0-2020.04.1-x86_64-linux-ubuntu14}
SIM=${FDRA_SIM:-/home/lsl/wafer-compiler-for-cgra/simulator/chipyard/sims/verilator}
T=$FDRA/software/tests
EXE=$FDRA/benchmarks/test4/conv2d_3x3/cgra_execute_dual.c
[ -f "$EXE" ] || EXE=$FDRA/benchmarks/test4/conv2d_3x3/cgra_execute.c

$RISCV/bin/riscv64-unknown-elf-gcc -DPREALLOCATE=1 -DMULTITHREAD=1 -mcmodel=medany \
  -std=gnu99 -O2 -ffast-math -fno-common -fno-builtin-printf -march=rv64gc \
  -I"$HERE" -I"$HERE/../s2" -I$T/riscv-tests -I$T/riscv-tests/env -I$T -I$T/include \
  -I$T/riscv-tests/benchmarks/common -nostdlib -nostartfiles -static \
  -T $T/riscv-tests/benchmarks/common/test.ld -DBAREMETAL=1 \
  "$HERE/main_conv_timed.c" "$EXE" \
  $T/riscv-tests/benchmarks/common/*.c $T/riscv-tests/benchmarks/common/*.S \
  -lm -lgcc -o "$HERE/conv.riscv"

(cd "$SIM" && timeout 900 ./simulator-chipyard-TramSoCConfig "$HERE/conv.riscv" \
  > "$HERE/conv_run.log" 2>&1) || true
grep -h CONV_PERF "$HERE/conv_run.log" || { echo 'run produced no CONV_PERF'; exit 1; }
python3 "$HERE/../s2/check_conv2d.py" "$HERE/conv_run.log"
