from dataclasses import dataclass

from xdsl.dialects import builtin, affine, func, arith, memref, linalg
from xdsl.dialects.experimental import dataflow
#from xdsl.transforms.experimental.dataflow_to_func import ConvertDataflowToFunc

from xdsl import traits
from xdsl.ir import (
    Block,
    BlockArgument,
    MLContext,
    Operation,
    OpResult,
    Region
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from xdsl.builder import Builder

@dataclass
class PrepareTop(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        if op.sym_name.data == "forward":
            top = dataflow.Top.from_function(op)

            for top_op in top.walk():
                if isinstance(top_op, affine.For) and not isinstance(top_op.parent.parent_op(), affine.For):
                    top_op.attributes["outer"] = builtin.i1

            rewriter.replace_matched_op(top)

def is_in_loop_context(loop, op):
    if isinstance(op, affine.For):
        current_op = op.parent_op()
    elif isinstance(op, BlockArgument):
        current_op = op.owner.parent.parent
    elif isinstance(op, OpResult):
        current_op = op.owner.parent.parent.parent
        while not isinstance(current_op, Operation):
            current_op = current_op.parent
    else:
        return False

    if current_op == loop:
        return True
    else:
        return is_in_loop_context(loop, current_op)



def get_ops_operands_map(op, ops_operands_map: dict, parent_loop: affine.For):
    # Generate a map of all the operations in the loop with operands defined outside
    # the loop to these operands and the index they will occupy in the wrapping function
    # (this is the order in which they've been discovered) 
    # ops_operands_map[operation] = [[operand_x, idx_x], ..., [operand_y, idx_y]]. The index
    # will be 'clone' if the operation should be cloned inside the function (constants).

    for body_op in op.body.block.ops:
        if isinstance(body_op, affine.For):
            get_ops_operands_map(body_op, ops_operands_map, parent_loop)
        else:
            for idx, operand in enumerate(body_op.operands):
                operand_in_context = is_in_loop_context(parent_loop, operand)
                if not operand_in_context:
                    if body_op not in ops_operands_map:
                        ops_operands_map[body_op] = []

                    if isinstance(operand, OpResult) and isinstance(operand.owner, arith.Constant):
                        ops_operands_map[body_op].append([operand, 'clone'])
                    else:
                        ops_operands_map[body_op].append([operand, idx])

@dataclass
class PromoteLoopToFuncs(RewritePattern):
    func_args_map: dict
    func_idx = iter(range(1024))

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.For, rewriter: PatternRewriter, /):
        if "outer" in op.attributes:
            ops_operands_map = dict()

            # The operands of the memory operations in the loop will become the new function arguments.
            # The operations will become the users of the corresponding block arguments
            get_ops_operands_map(op, ops_operands_map, op)

            @Builder.region
            def loop_region(builder: Builder):
                builder.insert(func.Return())

            # The constant operands will be excluced from the signature and copied inside the function instead
            all_operands = []
            for k in ops_operands_map.keys():
                for operand_list in ops_operands_map[k]:
                    if operand_list[1] != 'clone':
                        all_operands.append(operand_list[0])

            input_types = []

            # Generate the input types from all the operands without repetition. Generate map of each operand 
            # to the index of the corresponding function argument
            node_name = f"node_{next(self.func_idx)}"
            self.func_args_map[node_name] = []

            operand_to_func_arg_idx_map = dict()
            visited_operands = set()
            for idx, operand in enumerate(all_operands):
                if operand not in visited_operands:
                    input_types.append(operand.type)

                    visited_operands.add(operand)
                    operand_to_func_arg_idx_map[operand] = idx

                    self.func_args_map[node_name].append(operand)

            loop_func = func.FuncOp.from_region(node_name, [*input_types], [], loop_region)
            loop_func.attributes["node"] = builtin.i1

            for i in range(len(input_types)):
                rewriter.insert_block_argument(loop_region.block, i, input_types[i])

            # Define the use of each function argument to the corresponding operand in the loop operations. If the 
            # operand was a constant then it is cloned inside the function and used instead.
            operands_to_clone = set()
            for k in ops_operands_map.keys():
                for operand,idx in ops_operands_map[k]:
                    if idx == "clone":
                        operands_to_clone.add(operand)

            constant_copies = []
            for func_arg_user in ops_operands_map:
                for idx, operand in enumerate(func_arg_user.operands):
                    if operand in operands_to_clone:
                        clone_op = operand.owner.clone()
                        constant_copies.append(clone_op)
                        func_arg_user.operands[idx] = clone_op.result

                    elif operand in operand_to_func_arg_idx_map:
                        func_arg_idx = operand_to_func_arg_idx_map[operand]
                        func_arg_user.operands[idx] = loop_region.block.args[func_arg_idx]

            rewriter.insert_op_before_matched_op(loop_func)
            op.detach()

            rewriter.insert_op_before(op, loop_region.blocks[0].first_op)
            for constant_copy in constant_copies:
                rewriter.insert_op_before(constant_copy, loop_region.blocks[0].first_op)

def find_module(op):
    module = op
    while not isinstance(module, builtin.ModuleOp):
        module = module.parent_op()

    return module

@dataclass
class ReplaceFuncDeclarationsWithCalls(RewritePattern):
    func_args_map: dict

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Top, rewriter: PatternRewriter, /):
        module = find_module(op)

        for insert_point in module.body.block.ops:
            if isinstance(insert_point, func.FuncOp):
                break


        for body_op in op.body.block.ops:
            if isinstance(body_op, func.FuncOp):
                node_name = body_op.sym_name.data
                node_call = func.Call(node_name, self.func_args_map[node_name], [])

                rewriter.insert_op_before(node_call, body_op)
                body_op.detach()
                rewriter.insert_op_before(body_op, insert_point)


@dataclass
class DeadCodeElimination(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Top, rewriter: PatternRewriter, /):
        for body_op in op.body.block.ops:
            if isinstance(body_op, arith.Constant) and not body_op.result.uses:
                body_op.detach()
                body_op.erase()


@dataclass
class PromoteFuncsToNodes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if "node" in op.attributes:
            module = find_module(op)

            node_name = op.sym_name.data
            node_body = rewriter.move_region_contents_to_new_regions(op.body)
            node = dataflow.Node(node_name, op.function_type, node_body)

            rewriter.insert_op_before_matched_op(node)
            rewriter.erase_matched_op()

            # loads stay as inputs and stores become outputs
            inputs = []
            outputs = []
            idx_in_out = [] # encodes which function arguments are inputs or outputs by index
            for arg in node.body.block.args:
                directions_list = []
                for use in arg.uses:
                    if isinstance(use.operation, affine.Load):
                        inputs.append(arg)
                        directions_list.append('in')
                    elif isinstance(use.operation, affine.Store):
                        outputs.append(arg)
                        directions_list.append('out')

                # If the arguments is used both to read and to write qualify it with 'inout'
                direction = directions_list[0]
                for other_direction in directions_list[1:]:
                    if other_direction != direction:
                        direction = 'inout'
                        break

                idx_in_out.append(direction)

            node_type = builtin.FunctionType.from_lists([in_arg.type for in_arg in inputs], [out_arg.type for out_arg in outputs])

            for node_op in node.body.block.ops:
                if isinstance(node_op, func.Return):
                    rewriter.replace_op(node_op, dataflow.NodeEnd())

            node.node_type = node_type

            # split arguments to the function call between inputs and outputs (specify in an attribute). Promote to node_call
            for mod_op in module.body.block.ops:
                if isinstance(mod_op, dataflow.Top):
                    for top_op in mod_op.body.block.ops:
                        if isinstance(top_op, func.Call) and top_op.callee.root_reference == op.sym_name:
                            in_out_attr = builtin.ArrayAttr(list(map(lambda x: builtin.StringAttr(x), idx_in_out)))

                            node_call = dataflow.NodeCall(node_name, top_op.arguments, [], in_out_attr)
                            rewriter.replace_op(top_op, node_call)

@dataclass
class GenerateDataflowGraph(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Top, rewriter: PatternRewriter, /):
        for body_op in op.body.block.ops:
            if isinstance(body_op, dataflow.NodeCall):
                node_call = body_op
                connected = dataflow.Connected(node_call)

                rewriter.insert_op_after(connected, node_call)

@dataclass
class DfCallCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.NodeCall, rewriter: PatternRewriter):
        rewriter.erase_matched_op()

@dataclass
class DeallocCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Dealloc, rewriter: PatternRewriter):
        rewriter.erase_matched_op()

@dataclass
class IdentifyTransposeAndInitNodes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Node, rewriter: PatternRewriter):
        ops_inside_loops = [node_op for node_op in op.walk() if not isinstance(node_op, affine.For) and not isinstance(node_op, dataflow.NodeEnd) and not isinstance(node_op, affine.Yield) and not isinstance(node_op, dataflow.Node)]
        if len(ops_inside_loops) == 2:
            if isinstance(ops_inside_loops[0], affine.Load) and isinstance(ops_inside_loops[1], affine.Store): # Condition for transpose node
                op.attributes["transpose_node"] = builtin.i1
            if isinstance(ops_inside_loops[0], arith.Constant) and isinstance(ops_inside_loops[1], affine.Store): # Condition for init node
                op.attributes["init_node"] = builtin.i1

@dataclass
class TransposePass(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, transpose_node: dataflow.Node, rewriter: PatternRewriter):
        if "transpose_node" in transpose_node.attributes:
            module = find_module(transpose_node)
            connected_copy = [mod_op for mod_op in module.walk() if isinstance(mod_op, dataflow.Connected) and mod_op.node.root_reference == transpose_node.sym_name][0]

            input_shape = connected_copy.operands[0].type.shape
            output_shape = connected_copy.operands[1].type.shape

            # Find the permutation array. Note that some of the dimensions will be equal, hence it is necessary to keep their coordinates
            # in a list, otherwise only the rightmost dimension would be recorded
            input_dim_to_coordinate = dict()
            for coord,dim in enumerate(input_shape):
                if not dim.data in input_dim_to_coordinate:
                    input_dim_to_coordinate[dim.data] = [coord]
                else:
                    input_dim_to_coordinate[dim.data].append(coord)

            permutation = []
            for coord,dim in enumerate(output_shape):
                orig_coord_for_dim = input_dim_to_coordinate[dim.data][0]
                del input_dim_to_coordinate[dim.data][0]
                permutation.append(orig_coord_for_dim)

            permutation = builtin.DenseArrayBase.from_list(builtin.i32, permutation)

            in_memref = connected_copy.operands[0]
            out_memref = connected_copy.operands[1]

            transpose = linalg.TransposeOp(in_memref, out_memref, permutation, builtin.TensorType(builtin.f32, output_shape))

            # Update the list of in nodes and the edges in the out node after removing the transpose node. Since these lists contain symbols instead of SSA values, this 
            # has to be done manually
            for out_node in connected_copy.out_nodes:
                connected_out = [mod_op for mod_op in module.walk() if isinstance(mod_op, dataflow.Connected) and mod_op.node.root_reference == out_node.root_reference][0]

                remove_node_from_connected(transpose_node, connected_out, module)

            rewriter.insert_op_before(transpose, connected_copy)
            rewriter.erase_op(connected_copy)
            rewriter.erase_matched_op()

def remove_node_from_connected(src_node : dataflow.Node, dst_connected : dataflow.Connected, module : builtin.ModuleOp):
    dst_in_nodes_lst = list(dst_connected.in_nodes.data)
    for dst_node_in_node_idx,dst_node_in_node in enumerate(dst_in_nodes_lst):
        if dst_node_in_node.root_reference == src_node.sym_name:
            del dst_in_nodes_lst[dst_node_in_node_idx]
    dst_connected.in_nodes = builtin.ArrayAttr(dst_in_nodes_lst)

    dst_edges = dict(dst_connected.edges.data)
    for k in dst_edges:
        dst_edges[k] = list(dst_edges[k])
    for k in dst_connected.edges.data:
        for dst_edge_idx,dst_edge_node_lst in enumerate(dst_connected.edges.data[k]):
            if list(dst_edge_node_lst)[0].root_reference == src_node.sym_name:
                del dst_edges[k][dst_edge_idx]

    for k in dst_edges:
        dst_edges[k] = builtin.ArrayAttr(dst_edges[k])
    dst_connected.edges = builtin.DictionaryAttr(dst_edges)

@dataclass
class IntegrateInitNodes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, init_node: dataflow.Node, rewriter: PatternRewriter):
        # There are two cases: init nodes that are directly connected to computational nodes and init nodes that are connected through a copy. The 
        # nodes that are connected through a copy don't have in or out nodes. In the second case it is necessary to traverse the copy and eventually 
        # remove it. Similarly, the subviews, which are not used, should be removed
        if "init_node" in init_node.attributes:
            module = find_module(init_node)
            connected_init = [mod_op for mod_op in module.walk() if isinstance(mod_op, dataflow.Connected) and mod_op.node.root_reference == init_node.sym_name][0]

            if connected_init.out_nodes:
                connected_out_lst = [use.operation for use in connected_init.operands[0].uses if isinstance(use.operation, dataflow.Connected) and use.operation != init_node]
            else:
                copy_op = [use.operation for use in connected_init.operands[0].uses if isinstance(use.operation, memref.CopyOp)][0]
                connected_out_lst = [use.operation for use in copy_op.operands[1].uses if isinstance(use.operation, dataflow.Connected)]
            
            connected_out_lst = sorted(connected_out_lst, key=lambda x: x.node.root_reference.data)

            for connected_out in connected_out_lst:
                if not connected_init.out_nodes:
                    for store_idx,store_operand in enumerate(connected_out.operands):
                        if store_operand == copy_op.operands[1]:
                            connected_out.operands[store_idx] = connected_init.operands[0]
                            break

                    subview_copy = [use.operation for use in copy_op.operands[1].uses if isinstance(use.operation, memref.Subview)]
                    if subview_copy:
                        subview_copy = subview_copy[0]
                        copy_after_subview = [use.operation for use in subview_copy.result.uses][0]
                        assert isinstance(copy_after_subview, memref.CopyOp)
                        rewriter.erase_op(copy_after_subview)
                        rewriter.erase_op(subview_copy)
                else:
                    for store_idx,store_operand in enumerate(connected_out.operands):
                        if store_operand == connected_init.operands[0]:
                            connected_out.operands[store_idx] = connected_init.operands[0]
                            break

                user_node = builtin.SymbolTable.lookup_symbol(init_node, connected_out.node)
                init_node_clone = init_node.clone()
                for init_op in init_node_clone.walk():
                    if isinstance(init_op, affine.Store):
                        init_op.operands[1] = user_node.body.block.args[store_idx]

                init_node_clone_ops = [init_op for init_op in init_node_clone.body.block.ops if not isinstance(init_op, dataflow.NodeEnd)]
                for init_clone_op in reversed(init_node_clone_ops):
                    init_clone_op.detach()
                    rewriter.insert_op_at_start(init_clone_op, user_node.body.block)

                if connected_init.out_nodes:
                    # Remove the init node from the in nodes list in the user node and the edges list
                    remove_node_from_connected(init_node, connected_out, module)
                rewriter.erase_op(connected_out)

            if not connected_init.out_nodes:
                rewriter.erase_op(copy_op)

            rewriter.erase_matched_op()

@dataclass
class PartitionNodeByForLoop(RewritePattern):
    func_idx = iter(range(1024))

    @op_type_rewrite_pattern
    def match_and_rewrite(self, node: dataflow.Node, rewriter: PatternRewriter):
        for_loops = [node_op for node_op in node.body.walk() if isinstance(node_op, affine.For)]

        for for_op in reversed(for_loops):
            loop_node = dataflow.Node(f"loop_node_{next(self.func_idx)}", builtin.FunctionType.from_lists([], []), Region())
            rewriter.insert_op_before(loop_node, for_op)

            for_op.detach()
            @Builder.region
            def loop_region(builder: Builder):
                builder.insert(for_op)
            rewriter.inline_region_at_start(loop_region, loop_node.regions[0])


        loops_nodes = [node_op for node_op in node.body.walk() if isinstance(node_op, dataflow.Node)]
        call_args = dict()
        for loop_node in reversed(loops_nodes):
            # Calculate inouts - these are the SSA values that are out of the context of the node but used 
            # by its operations
            node_inouts = []
            added_inout = set()
            for loop_inner_op in loop_node.walk():
                for inner_op_operand in loop_inner_op.operands:

                    if not loop_node.is_ancestor(inner_op_operand.owner) and not inner_op_operand in added_inout:
                        node_inouts.append(inner_op_operand)
                        added_inout.add(inner_op_operand)

            # Add inouts to the node (used by the operations inside)
            call_args[loop_node.sym_name] = []
            for node_inout in reversed(node_inouts):
                call_args[loop_node.sym_name].append(node_inout)

                rewriter.insert_block_argument(loop_node.body.block, 0, node_inout.type)

                loop_inner_ops_arg0_uses = []
                for inner_use in node_inout.uses:
                    loop_inner_ops_arg0_uses.append(inner_use)

                for arg0_use in loop_inner_ops_arg0_uses:
                    arg0_use.operation.operands[arg0_use.index] = loop_node.body.block.args[0]

            loop_node_call = dataflow.NodeCall(loop_node.sym_name.data, reversed(call_args[loop_node.sym_name]), [], builtin.ArrayAttr([]))
            rewriter.insert_op_before(loop_node_call, loop_node)
            loop_node.detach()
            rewriter.insert_op_before(loop_node, node)

        # Place node end operations at the end of the node operations
        for loop_node in loops_nodes:
            for loop_node_op in loop_node.walk():
                if isinstance(loop_node_op, dataflow.NodeEnd):
                    rewriter.erase_op(loop_node_op)

        for loop_node in loops_nodes:
            for loop_node_op in loop_node.walk():
                if isinstance(loop_node_op, dataflow.Node):
                    rewriter.insert_op_at_end(dataflow.NodeEnd(), loop_node_op.body.block)



@dataclass
class DataflowGraph(ModulePass):
    name = "dataflow-graph"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        func_args_map = dict()
        prepare_top_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PrepareTop(),
                ]
            ),
            apply_recursively=False,
        )
        prepare_top_pass.rewrite_module(op)

        #promote_loop_to_func_pass = PatternRewriteWalker(
        #    GreedyRewritePatternApplier(
        #        [
        #            PromoteLoopToFuncs(func_args_map),
        #        ]
        #    ),
        #    apply_recursively=False,
        #)
        #promote_loop_to_func_pass.rewrite_module(op)

        #extract_func_pass = PatternRewriteWalker(
        #    GreedyRewritePatternApplier(
        #        [
        #            ReplaceFuncDeclarationsWithCalls(func_args_map),
        #        ]
        #    ),
        #    apply_recursively=False,
        #)
        #extract_func_pass.rewrite_module(op)

        #dce_pass = PatternRewriteWalker(
        #    GreedyRewritePatternApplier(
        #        [
        #            DeadCodeElimination()
        #        ]
        #    ),
        #    apply_recursively=False,
        #)
        #dce_pass.rewrite_module(op)

        ##df_dialect_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            PromoteFuncsToNodes(),
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ##df_dialect_pass.rewrite_module(op)

        ##df_graph_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            GenerateDataflowGraph(),
        ##            DfCallCleanup(),
        ##            #DeallocCleanup()
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ##df_graph_pass.rewrite_module(op)


        ##identify_transpose_nodes_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            IdentifyTransposeAndInitNodes()
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ##identify_transpose_nodes_pass.rewrite_module(op)

        ##transpose_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            TransposePass()
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ###transpose_pass.rewrite_module(op)

        ##integrateinitnodes_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            IntegrateInitNodes()        
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ###integrateinitnodes_pass.rewrite_module(op)

        ##partition_by_for_loop_pass = PatternRewriteWalker(
        ##    GreedyRewritePatternApplier(
        ##        [
        ##            PartitionNodeByForLoop()
        ##        ]
        ##    ),
        ##    apply_recursively=False,
        ##)
        ##partition_by_for_loop_pass.rewrite_module(op)

        ##ConvertDataflowToFunc().apply(ctx, op)