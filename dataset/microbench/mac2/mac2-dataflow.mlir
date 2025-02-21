module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @d : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @c : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @b : memref<1xmemref<?xi32>> = uninitialized
  memref.global "private" @a : memref<1xmemref<?xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    cgra.dispatch {
      %0 = memref.get_global @a : memref<1xmemref<?xi32>>
      %1 = memref.get_global @b : memref<1xmemref<?xi32>>
      %2 = memref.get_global @c : memref<1xmemref<?xi32>>
      %3 = memref.get_global @d : memref<1xmemref<?xi32>>
      %4 = affine.load %0[0] : memref<1xmemref<?xi32>>
      %5 = affine.load %1[0] : memref<1xmemref<?xi32>>
      %6 = affine.load %2[0] : memref<1xmemref<?xi32>>
      %7 = affine.load %3[0] : memref<1xmemref<?xi32>>
      cgra.task {
        affine.for %arg2 = 1 to 99 {
          cgra.dispatch {
            %8 = affine.load %4[%arg2] : memref<?xi32>
            %9 = affine.load %5[%arg2] : memref<?xi32>
            %10 = arith.muli %8, %9 : i32
            %11 = affine.load %arg0[0] : memref<?xi32>
            %12 = arith.addi %11, %10 : i32
            affine.store %12, %arg0[0] : memref<?xi32>
            %13 = affine.load %4[%arg2] : memref<?xi32>
            %14 = affine.load %5[%arg2] : memref<?xi32>
            %15 = arith.addi %14, %c1_i32 : i32
            %16 = arith.muli %13, %15 : i32
            %17 = affine.load %6[%arg2] : memref<?xi32>
            %18 = arith.muli %16, %17 : i32
            %19 = affine.load %7[%arg2] : memref<?xi32>
            %20 = arith.muli %18, %19 : i32
            %21 = affine.load %arg1[0] : memref<?xi32>
            %22 = arith.addi %21, %20 : i32
            affine.store %22, %arg1[0] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}

