module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @b : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @a : memref<1xmemref<?xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = memref.get_global @a : memref<1xmemref<?xi32>>
    %1 = memref.get_global @b : memref<1xmemref<?xi32>>
    affine.for %arg1 = 1 to 99 {
      %2 = affine.load %0[0] : memref<1xmemref<?xi32>>
      %3 = affine.load %2[%arg1] : memref<?xi32>
      %4 = arith.muli %3, %c2_i32 : i32
      %5 = affine.load %2[%arg1 + 1] : memref<?xi32>
      %6 = arith.muli %5, %c2_i32 : i32
      %7 = arith.addi %4, %6 : i32
      %8 = affine.load %1[0] : memref<1xmemref<?xi32>>
      %9 = affine.load %8[%arg1] : memref<?xi32>
      %10 = arith.muli %9, %c2_i32 : i32
      %11 = affine.load %8[%arg1 + 3] : memref<?xi32>
      %12 = arith.muli %11, %c2_i32 : i32
      %13 = arith.addi %10, %12 : i32
      %14 = arith.muli %7, %13 : i32
      %15 = arith.muli %11, %c3_i32 : i32
      %16 = arith.muli %14, %15 : i32
      %17 = arith.muli %16, %9 : i32
      %18 = affine.load %arg0[0] : memref<?xi32>
      %19 = arith.addi %18, %17 : i32
      affine.store %19, %arg0[0] : memref<?xi32>
    }
    return
  }
}
