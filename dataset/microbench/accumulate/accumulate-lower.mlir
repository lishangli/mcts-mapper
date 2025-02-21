module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @b : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @a : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @c : memref<1xmemref<?xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    cgra.schedule(%arg0) : memref<?xi32> {
    ^bb0(%arg1: memref<?xi32>):
      %0 = memref.get_global @c : memref<1xmemref<?xi32>>
      %1 = memref.get_global @a : memref<1xmemref<?xi32>>
      %2 = memref.get_global @b : memref<1xmemref<?xi32>>
      %3 = affine.load %0[0] : memref<1xmemref<?xi32>>
      %4 = affine.load %1[0] : memref<1xmemref<?xi32>>
      %5 = affine.load %2[0] : memref<1xmemref<?xi32>>
      cgra.node(%4, %5, %3, %arg1) -> (%3, %arg1) {inputTaps = [0 : i32, 0 : i32, 0 : i32, 0 : i32], isLegal = true, pes = 3 : i32} : (memref<?xi32>, memref<?xi32>, memref<?xi32>, memref<?xi32>) -> (memref<?xi32>, memref<?xi32>) {
      ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>, %arg4: memref<?xi32>, %arg5: memref<?xi32>, %arg6: memref<?xi32>, %arg7: memref<?xi32>):
        affine.for %arg8 = 1 to 99 {
          %6 = affine.load %arg2[%arg8 + 1] : memref<?xi32>
          %7 = affine.load %arg3[%arg8 - 1] : memref<?xi32>
          %8 = arith.addi %6, %7 : i32
          %9 = affine.load %arg4[%arg8] : memref<?xi32>
          %10 = arith.muli %9, %8 : i32
          affine.store %10, %arg4[%arg8] : memref<?xi32>
          %11 = affine.load %arg5[0] : memref<?xi32>
          %12 = arith.addi %11, %10 : i32
          affine.store %12, %arg5[0] : memref<?xi32>
        }
      }
    }
    return
  }
}

