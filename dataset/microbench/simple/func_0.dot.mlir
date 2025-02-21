module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    cgra.schedule(%arg0) : memref<?xi32> {
    ^bb0(%arg1: memref<?xi32>):
      %c8_i32 = arith.constant 8 : i32
      %c7_i32 = arith.constant 7 : i32
      %c6_i32 = arith.constant 6 : i32
      %c5_i32 = arith.constant 5 : i32
      %c4_i32 = arith.constant 4 : i32
      %c3_i32 = arith.constant 3 : i32
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %alloca = memref.alloca() : memref<4xi32>
      %alloca_0 = memref.alloca() : memref<4xi32>
      affine.store %c1_i32, %alloca[0] : memref<4xi32>
      affine.store %c2_i32, %alloca[1] : memref<4xi32>
      affine.store %c3_i32, %alloca[2] : memref<4xi32>
      affine.store %c4_i32, %alloca[3] : memref<4xi32>
      affine.store %c5_i32, %alloca_0[0] : memref<4xi32>
      affine.store %c6_i32, %alloca_0[1] : memref<4xi32>
      affine.store %c7_i32, %alloca_0[2] : memref<4xi32>
      affine.store %c8_i32, %alloca_0[3] : memref<4xi32>
      cgra.node(%alloca, %alloca_0, %arg1) -> (%arg1) {inputTaps = [0 : i32, 0 : i32, 0 : i32], isLegal = true, pes = 2 : i32} : (memref<4xi32>, memref<4xi32>, memref<?xi32>) -> memref<?xi32> {
      ^bb0(%arg2: memref<4xi32>, %arg3: memref<4xi32>, %arg4: memref<?xi32>, %arg5: memref<?xi32>):
        affine.for %arg6 = 0 to 4 {
          %0 = affine.load %arg2[%arg6] : memref<4xi32>
          %1 = affine.load %arg3[%arg6] : memref<4xi32>
          %2 = arith.addi %0, %1 : i32
          %3 = affine.load %arg4[0] : memref<?xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %arg4[0] : memref<?xi32>
        }
      }
    }
    return
  }
}

