from dataclasses import dataclass

from xdsl.dialects import builtin, affine, func, arith, memref, linalg
from xdsl.dialects.experimental import dataflow

from xdsl import traits
from xdsl.ir import (
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
class NodeToFunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Node, rewriter: PatternRewriter):
        node_inputs = [block_arg.type for block_arg in op.body.block.args]

        func_region = rewriter.move_region_contents_to_new_regions(op.body)
        rewriter.replace_op(func_region.block.last_op, func.Return())

        node_func = func.FuncOp.from_region(op.sym_name.data, node_inputs, [], func_region)

        node_func.attributes["node_func"] = builtin.i1

        rewriter.replace_matched_op(node_func)

@dataclass
class ConnectedToNodeCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Connected, rewriter: PatternRewriter):
        node_call = func.Call(op.node, op.arguments, [])

        rewriter.replace_matched_op(node_call)

@dataclass
class TopToFunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Top, rewriter: PatternRewriter):
        node_inputs = [block_arg.type for block_arg in op.body.block.args]

        func_region = rewriter.move_region_contents_to_new_regions(op.body)
        rewriter.replace_op(func_region.block.last_op, func.Return())

        top_func = func.FuncOp.from_region(op.sym_name.data, node_inputs, [], func_region)

        rewriter.replace_matched_op(top_func)

@dataclass
class NodeCallToFuncCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.NodeCall, rewriter: PatternRewriter):
        node_call = func.Call(op.callee, op.arguments, [])

        rewriter.replace_matched_op(node_call)

CACHE_SIZE = 100

@dataclass
class CacheNodeInputs(RewritePattern):
    # Add a cache for the inputs and another for the outputs. For now the size fixed at size 100, but this should be adjusted 
    # to the input size in the future. Fill the caches with the inputs and the results of the computational loop and replace 
    # the operands of loads and stores with the corresponding caches.
    @op_type_rewrite_pattern
    def match_and_rewrite(self, node_func: func.FuncOp, rewriter: PatternRewriter):
        load_op = [node_func_op for node_func_op in node_func.walk() if isinstance(node_func_op, affine.Load)]
        store_op = [node_func_op for node_func_op in node_func.walk() if isinstance(node_func_op, affine.Store)]
        if load_op:
            for load_op_external, store_op_external in reversed(list(zip(load_op, store_op))):
                cache_load = memref.Alloca.get(builtin.Float32Type(), None, [CACHE_SIZE], None)
                cache_store = memref.Alloca.get(builtin.Float32Type(), None, [CACHE_SIZE], None)
                rewriter.insert_op_at_start(cache_load, node_func.body.block)
                rewriter.insert_op_after(cache_store, cache_load)

                @Builder.region([builtin.IndexType()])
                def cache_load_for_region(builder: Builder, args: tuple[BlockArgument, ...]):
                    load_internal_for_load = affine.Load(load_op_external.memref, args[0])
                    store_internal_for_load = affine.Store(load_internal_for_load.result, cache_load.memref, args[0])
                    builder.insert(load_internal_for_load)
                    builder.insert(store_internal_for_load)
                    builder.insert(affine.Yield.get())

                load_op_external.operands[0] = cache_load.memref

                cache_load_for = affine.For.from_region([], [], 0, CACHE_SIZE, cache_load_for_region)
                rewriter.insert_op_after(cache_load_for, cache_load)


                @Builder.region([builtin.IndexType()])
                def cache_store_for_region(builder: Builder, args: tuple[BlockArgument, ...]):
                    load_internal_for_store = affine.Load(cache_store.memref, args[0])
                    store_internal_for_store = affine.Store(load_internal_for_store.result, store_op_external.memref, args[0])
                    builder.insert(load_internal_for_store)
                    builder.insert(store_internal_for_store)
                    builder.insert(affine.Yield.get())

                store_op_external.operands[1] = cache_store.memref

                cache_store_for = affine.For.from_region([], [], 0, CACHE_SIZE, cache_store_for_region)
                rewriter.insert_op_before(cache_store_for, node_func.body.block.last_op)
@dataclass
class ConvertDataflowToFunc(ModulePass):
    name = "dataflow-to-func"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        dataflow_to_func_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    NodeToFunc(),
                    ConnectedToNodeCalls(),
                    TopToFunc(),
                    NodeCallToFuncCall()
                ]
            ),
            apply_recursively=False,
        )
        dataflow_to_func_pass.rewrite_module(op)

        cache_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    CacheNodeInputs()
                ]
            ),
            apply_recursively=False,
        )
        #cache_pass.rewrite_module(op)