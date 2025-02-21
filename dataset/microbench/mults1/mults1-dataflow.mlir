module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" @a : memref<1xmemref<?xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10_i32 = arith.constant 10 : i32
    %c20_i32 = arith.constant 20 : i32
    %c39_i32 = arith.constant 39 : i32
    %c15_i32 = arith.constant 15 : i32
    cgra.dispatch {
      %0 = memref.get_global @a : memref<1xmemref<?xi32>>
      %1 = affine.load %0[0] : memref<1xmemref<?xi32>>
      cgra.task {
        affine.for %arg1 = 1 to 99 {
          cgra.dispatch {
            %2 = affine.load %1[%arg1] : memref<?xi32>
            %3 = arith.muli %2, %c10_i32 : i32
            %4 = affine.load %1[%arg1 + 1] : memref<?xi32>
            %5 = arith.muli %4, %c20_i32 : i32
            %6 = arith.addi %3, %5 : i32
            %7 = affine.load %1[%arg1 + 2] : memref<?xi32>
            %8 = arith.muli %7, %c39_i32 : i32
            %9 = arith.addi %6, %8 : i32
            %10 = affine.load %1[%arg1 + 3] : memref<?xi32>
            %11 = arith.muli %10, %c15_i32 : i32
            %12 = arith.addi %9, %11 : i32
            %13 = affine.load %arg0[0] : memref<?xi32>
            %14 = arith.addi %13, %12 : i32
            affine.store %14, %arg0[0] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}

