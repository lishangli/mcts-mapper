# Why the dense flow stops at ~3 ops/cycle, and what would move it

Status: design note, 2026-11-07. Every number below is measured and reproducible
with the commands in `docs/EXPERIMENTS.md`; anything projected is labelled.

## 1. Where the time goes (nn4x2, 4 inputs x 2 outputs = 16 ops/sample)

Stream sharing on, both capture channels, store clamped to 512 B/channel at 256
samples (round 21):

| samples | load | exec | store | total | ops | ops/cycle |
|---|---|---|---|---|---|---|
| 64 | 61 | 27 | 554 | 653 | 1024 | 1.57 |
| 128 | 122 | 6 | 756 | 915 | 2048 | **2.24** |
| 256 | 230 | 6 | 1121 | 1406 | 4096 | **2.91** |

(The 64-sample exec of 27 vs 6 is placement noise: the same kernel scheduled on
different PEs.)

Marginal cost between 128 and 256 samples, per sample:

    load  +108/128 = 0.84 cycles   (4 input words -> ~0.21 cycles/word)
    store +365/128 = 2.85 cycles   (2 output words -> ~1.43 cycles/word)
    total  3.84 cycles/sample

So the dense flow is **result-drain bound**, not input bound: an output word
costs about 1.4 cycles to write back, an input word about 0.2 cycles to load,
and the array itself (exec) takes 6 cycles for the whole kernel.

## 2. The same accounting for the convolution

`conv2d_3x3`: 195 cycles, 196 outputs, 1764 MACs, 3528 ops (`EXECUTION_MODEL.md`
§4b) — about **1.0 cycle per output word** and **18 ops per output word**.

## 3. One parameter: ops per output word, and how fast a word drains

    ops/cycle = (ops per output word) / (cycles per output word)

| kernel | ops per output word | cycles per output word | predicted | measured |
|---|---|---|---|---|
| nn4x2 @256 | 16 / 2 = 8 | 1.43 | 5.6 | **2.91** |
| conv2d_3x3 | 18 / 1 = 18 | 1.0 | 18 | **18.1** |

The conv row is exact. The dense row is off by ~2x, and the reason is the fixed
cost that the marginal does not include: at 256 samples the kernel still pays
config download, the window drain (~250 cycles per `EXECUTION_MODEL.md`) and the
launch itself, so its measured average is 2.91 against a 5.6 marginal.

Two independent levers follow, and only two:

1. **raise ops per output word** (dense: 8 today). For a layer with I inputs and
   O outputs it is `2 x I x O / O = 2 x I`, i.e. **it depends only on the number
   of inputs per output**, exactly like the conv's `2 x taps`. nn4x2 has I = 4;
   the conv has 9 taps.
2. **drain output words faster** (dense: 1.43 cycles/word vs conv's 1.0).

## 4. Why I cannot simply be raised today

`2 x I` requires 2I-1 compute nodes per output: nn4x2 (I = 4) is already 32 DFG
nodes and maps only sometimes (653 s to 2400 s, some placements never emit). A
16-input layer would be ~112 nodes, far past the measured ceiling of 20-24 nodes
(`docs/STATUS.md`). So the intensity lever is blocked by mapping capacity, not
by arithmetic.

## 5. What is left that is cheap

* **readback**: the store budget is ~1 KiB per launch (measured, round 21). Even
  a perfect kernel with O outputs can only return 256/O samples, so larger I
  makes the launch cost harder to amortise, not easier.
* **time multiplexing**: `S2_PACK=1` folds independent ops onto one PE, one per
  slot. If it also lets one operand word feed several outputs, the ops-per-output
  ratio rises without new nodes. This is the only lever that is both plausible
  and available today.

## 6. The smallest falsifying experiment

Run `nn4x2` with `S2_PACK=1`:

1. does it still map and stay bit-exact `[18, 21]`?
2. do the cycles fall, or does folding only trade PEs for latency?

**Result (measured, round 23): it does not map.** `nn4x2` with `S2_PACK=1`
gave 0/1 after 1862 s over five remap attempts, last reason `no const IOB`.
`S2_PACK` also disables `S2_NO_PHASE`, so every operand must land on an exact
context residue, and a stream IOB cannot satisfy that here. Folding after the
fact is therefore incompatible with the context constraint in the current
router — an II schedule would have to assign contexts *together with* routing,
which is a scheduler rewrite, not a flag.

So ops-per-output-word is capped by the drain rate in this flow, and ~3 ops/cycle
for dense is its honest ceiling.

## 7. What would have to be true to close the gap

For a dense layer to reach 18 ops/cycle with I inputs and O outputs:

    (2 x I x O) / (O x c_store + I x c_load) >= 18     with c_store = 1.43, c_load = 0.21

which needs `I >= 9` at O large enough to amortise, i.e. the same 9-way reuse the
convolution gets from its taps — **and** mapping capacity for ~2I-1 nodes per
output, **and** a readback of 256/O samples. None of the three is currently
available; the numbers above say which one binds first in each case.
