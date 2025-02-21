module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @a : memref<1xmemref<?xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    cgra.schedule(%arg0) : memref<?xi32> {
    ^bb0(%arg1: memref<?xi32>):
      %0 = memref.get_global @a : memref<1xmemref<?xi32>>
      %1 = affine.load %0[0] : memref<1xmemref<?xi32>>
      cgra.node(%1) -> (%arg1) {isLegal = true, pes = 5 : i32} : (memref<?xi32>) -> memref<?xi32> {
      ^bb0(%arg2: memref<?xi32>, %arg3: memref<?xi32>):
        affine.for %arg4 = 1 to 99 {
          %c3_i32 = arith.constant 3 : i32
          %c20_i32 = arith.constant 20 : i32
          %c10_i32 = arith.constant 10 : i32
          %2 = affine.load %arg2[%arg4] : memref<?xi32>
          %3 = arith.muli %2, %c10_i32 : i32
          %4 = affine.load %arg2[%arg4 + 1] : memref<?xi32>
          %5 = arith.muli %4, %c20_i32 : i32
          %6 = arith.addi %3, %5 : i32
          %7 = affine.load %arg2[%arg4 + 2] : memref<?xi32>
          %8 = arith.muli %7, %c3_i32 : i32
          %9 = arith.addi %6, %8 : i32
          affine.store %9, %arg3[%arg4] : memref<?xi32>
        }
      }
    }
    return
  }
}

