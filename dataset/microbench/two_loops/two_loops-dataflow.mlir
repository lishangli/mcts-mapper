module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @func(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c20_i32 = arith.constant 20 : i32
    %c-1_i32 = arith.constant -1 : i32
    cgra.dispatch {
      %alloca = memref.alloca() : memref<9xi32>
      cgra.task {
        affine.for %arg2 = 0 to 9 {
          cgra.dispatch {
            %0 = affine.load %alloca[%arg2] : memref<9xi32>
            %1 = arith.addi %0, %c2_i32 : i32
            %2 = arith.muli %1, %1 : i32
            %3 = arith.addi %2, %1 : i32
            %4 = arith.addi %3, %c-1_i32 : i32
            %5 = arith.muli %4, %c3_i32 : i32
            %6 = arith.addi %5, %c1_i32 : i32
            %7 = affine.load %arg0[0] : memref<?xi32>
            %8 = arith.addi %7, %6 : i32
            affine.store %8, %arg0[0] : memref<?xi32>
          }
        }
      }
      cgra.task attributes {task_id = 1 : i32} {
        affine.for %arg2 = 0 to 9 {
          cgra.dispatch {
            %0 = affine.load %alloca[%arg2] : memref<9xi32>
            %1 = arith.addi %0, %c2_i32 : i32
            %2 = arith.muli %1, %1 : i32
            %3 = arith.addi %2, %1 : i32
            %4 = arith.addi %3, %c-1_i32 : i32
            %5 = arith.muli %4, %c3_i32 : i32
            %6 = arith.addi %5, %c3_i32 : i32
            %7 = arith.muli %6, %6 : i32
            %8 = arith.muli %7, %6 : i32
            %9 = arith.addi %8, %7 : i32
            %10 = arith.addi %9, %c20_i32 : i32
            %11 = arith.muli %10, %c2_i32 : i32
            %12 = affine.load %arg1[0] : memref<?xi32>
            %13 = arith.addi %12, %11 : i32
            affine.store %13, %arg1[0] : memref<?xi32>
          }
        }
      }
    }
    return
  }
}

