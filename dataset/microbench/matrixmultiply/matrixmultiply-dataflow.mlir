module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global @resultAB1 : memref<20x20xi32> = uninitialized
  memref.global "private" @B1 : memref<1xmemref<20xi32>> = uninitialized
  memref.global "private" @A1 : memref<1xmemref<20xi32>> = uninitialized
  func.func @func(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    cgra.dispatch {
      %0 = memref.get_global @A1 : memref<1xmemref<20xi32>>
      %1 = memref.get_global @B1 : memref<1xmemref<20xi32>>
      %2 = memref.get_global @resultAB1 : memref<20x20xi32>
      %3 = affine.load %0[0] : memref<1xmemref<20xi32>>
      %4 = affine.load %1[0] : memref<1xmemref<20xi32>>
      cgra.task {
        affine.for %arg1 = 0 to 20 {
          affine.for %arg2 = 0 to 20 {
            affine.for %arg3 = 0 to 20 {
              cgra.dispatch {
                %5 = affine.load %3[%arg3 + %arg1] : memref<20xi32>
                %6 = affine.load %4[%arg2 + %arg3] : memref<20xi32>
                %7 = arith.muli %5, %6 : i32
                %8 = affine.load %arg0[0] : memref<?xi32>
                %9 = arith.addi %8, %7 : i32
                affine.store %9, %arg0[0] : memref<?xi32>
                affine.store %9, %2[%arg1, %arg2] : memref<20x20xi32>
              }
            }
          }
        }
      }
    }
    return
  }
}

