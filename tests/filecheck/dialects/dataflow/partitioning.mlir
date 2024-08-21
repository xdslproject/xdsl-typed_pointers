// RUN: xdsl-opt -p dataflow-graph %s | filecheck %s

builtin.module {
    "func.func"() ({
            ^bb0(%in: memref<100xf32>, %out: memref<100xf32>):
        %out_A = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100xf32>
        %out_D = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100xf32>

        // D
        "affine.for"() ({
        ^bb0(%arg0 : index):
            "affine.for"() ({
                    ^bb0(%arg1 : index):
                            %5 = "affine.load"(%out_A, %arg1) {"map" = affine_map<(d0) -> (d0 mod 100)>} : (memref<100xf32>, index) -> f32
                            %arg0_i = "arith.index_cast"(%arg1) : (index) -> (i32)
                            %arg0_fp = "arith.sitofp"(%arg0_i) : (i32) -> f32
                            %6 = arith.addf %5, %arg0_fp : f32
                            "affine.store"(%6, %out_D, %arg1) {"map" = affine_map<(d0) -> (d0 mod 100)>} : (f32, memref<100xf32>, index) -> ()
                            "affine.yield"() : () -> ()
            }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (50)>, "step" = 1 : index} : () -> ()
            "affine.for"() ({
                  ^bb0(%arg2 : index):
                          %6 = "affine.load"(%out_A, %arg2) {"map" = affine_map<(d0) -> (d0 mod 100)>} : (memref<100xf32>, index) -> f32
                          %arg0_i = "arith.index_cast"(%arg2) : (index) -> (i32)
                          %arg0_fp = "arith.sitofp"(%arg0_i) : (i32) -> f32
                          %7 = arith.subf %6, %arg0_fp : f32
                          "affine.store"(%7, %out_D, %arg2) {"map" = affine_map<(d0) -> (d0 mod 100)>} : (f32, memref<100xf32>, index) -> ()
                          "affine.yield"() : () -> ()
            }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (50)>, "step" = 1 : index} : () -> ()
            "affine.yield"() : () -> ()
        }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (16)>, "step" = 1 : index} : () -> ()

        "func.return"() : () -> ()
        }) {"function_type" = (memref<100xf32>, memref<100xf32>) -> (), "sym_name" = "forward"} : () -> ()
}

//CHECK: builtin.module {
//CHECK-NEXT:   df.node @loop_node_0(%0 : memref<100xf32>, %1 : memref<100xf32>) {
//CHECK-NEXT:     "affine.for"() ({
//CHECK-NEXT:     ^0(%arg2 : index):
//CHECK-NEXT:       %2 = "affine.load"(%0, %arg2) {"map" = affine_map<(d0) -> ((d0 mod 100))>} : (memref<100xf32>, index) -> f32
//CHECK-NEXT:       %arg0_i = "arith.index_cast"(%arg2) : (index) -> i32
//CHECK-NEXT:       %arg0_fp = "arith.sitofp"(%arg0_i) : (i32) -> f32
//CHECK-NEXT:       %3 = arith.subf %2, %arg0_fp : f32
//CHECK-NEXT:       "affine.store"(%3, %1, %arg2) {"map" = affine_map<(d0) -> ((d0 mod 100))>} : (f32, memref<100xf32>, index) -> ()
//CHECK-NEXT:       "affine.yield"() : () -> ()
//CHECK-NEXT:     }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (50)>, "step" = 1 : index} : () -> ()
//CHECK-NEXT:     df.node_end
//CHECK-NEXT:   }
//CHECK-NEXT:   df.node @loop_node_1(%4 : memref<100xf32>, %5 : memref<100xf32>) {
//CHECK-NEXT:     "affine.for"() ({
//CHECK-NEXT:     ^1(%arg1 : index):
//CHECK-NEXT:       %6 = "affine.load"(%4, %arg1) {"map" = affine_map<(d0) -> ((d0 mod 100))>} : (memref<100xf32>, index) -> f32
//CHECK-NEXT:       %arg0_i_1 = "arith.index_cast"(%arg1) : (index) -> i32
//CHECK-NEXT:       %arg0_fp_1 = "arith.sitofp"(%arg0_i_1) : (i32) -> f32
//CHECK-NEXT:       %7 = arith.addf %6, %arg0_fp_1 : f32
//CHECK-NEXT:       "affine.store"(%7, %5, %arg1) {"map" = affine_map<(d0) -> ((d0 mod 100))>} : (f32, memref<100xf32>, index) -> ()
//CHECK-NEXT:       "affine.yield"() : () -> ()
//CHECK-NEXT:     }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (50)>, "step" = 1 : index} : () -> ()
//CHECK-NEXT:     df.node_end
//CHECK-NEXT:   }
//CHECK-NEXT:   df.node @loop_node_2(%8 : memref<100xf32>, %9 : memref<100xf32>) {
//CHECK-NEXT:     "affine.for"() ({
//CHECK-NEXT:     ^2(%arg0 : index):
//CHECK-NEXT:       df.node_call @loop_node_1(%8, %9) {"args_directions" = []} : (memref<100xf32>, memref<100xf32>) -> ()
//CHECK-NEXT:       df.node_call @loop_node_0(%8, %9) {"args_directions" = []} : (memref<100xf32>, memref<100xf32>) -> ()
//CHECK-NEXT:       "affine.yield"() : () -> ()
//CHECK-NEXT:     }) {"lower_bound" = affine_map<() -> (0)>, "upper_bound" = affine_map<() -> (16)>, "step" = 1 : index} : () -> ()
//CHECK-NEXT:     df.node_end
//CHECK-NEXT:   }
//CHECK-NEXT:   df.node @node_0(%10 : memref<100xf32>, %11 : memref<100xf32>) -> (memref<100xf32>, memref<100xf32>) {
//CHECK-NEXT:     df.node_call @loop_node_2(%10, %11) {"args_directions" = []} : (memref<100xf32>, memref<100xf32>) -> ()
//CHECK-NEXT:     df.node_end
//CHECK-NEXT:   }
//CHECK-NEXT:   df.top @top(%in : memref<100xf32>, %out : memref<100xf32>) {
//CHECK-NEXT:     %out_A = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100xf32>
//CHECK-NEXT:     %out_D = "memref.alloc"() {"operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100xf32>
//CHECK-NEXT:     "df.connected"(%out_A, %out_D) {"node" = @node_0, "in_nodes" = [], "out_nodes" = [], "edges" = {}} : (memref<100xf32>, memref<100xf32>) -> ()
//CHECK-NEXT:     df.top_end
//CHECK-NEXT:   }
//CHECK-NEXT: }