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

        node_func = func.FuncOp.from_region(op.sym_name, node_inputs, [], func_region)

        node_func.attributes["node_func"] = builtin.i1

        rewriter.replace_matched_op(node_func)

@dataclass
class ConnectedToNodeCalls(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Connected, rewriter: PatternRewriter):
        node_call = func.Call(op.node, op.arguments, [])
        #print("NODE CALL: ", node_call)

        rewriter.replace_matched_op(node_call)

@dataclass
class TopToFunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.Top, rewriter: PatternRewriter):
        node_inputs = [block_arg.type for block_arg in op.body.block.args]

        func_region = rewriter.move_region_contents_to_new_regions(op.body)
        rewriter.replace_op(func_region.block.last_op, func.Return())

        top_func = func.FuncOp.from_region(op.sym_name, node_inputs, [], func_region)

        rewriter.replace_matched_op(top_func)

@dataclass
class NodeCallToFuncCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dataflow.NodeCall, rewriter: PatternRewriter):
        node_call = func.Call(op.callee, op.arguments, [])

        rewriter.replace_matched_op(node_call)

@dataclass
class CacheNodeInputs(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, node_func: func.FuncOp, rewriter: PatternRewriter):
        load_op = [node_func_op for node_func_op in node_func.walk() if isinstance(node_func_op, affine.Load)]
        if load_op:
            for load_op_external in reversed(load_op):
                cache = memref.Alloca.get(builtin.Float32Type(), None, [100], None)
                rewriter.insert_op_at_start(cache, node_func.body.block)

                @Builder.region([builtin.IndexType()])
                def cache_for_region(builder: Builder, args: tuple[BlockArgument, ...]):
                    load_internal = affine.Load(load_op_external.memref, args[0])
                    store_internal = affine.Store(load_internal.result, cache.memref, args[0])
                    builder.insert(load_internal)
                    builder.insert(store_internal)
                    builder.insert(affine.Yield.get())

                load_op_external.operands[0] = cache.memref

                cache_for = affine.For.from_region([], [], 0, 100, cache_for_region)
                rewriter.insert_op_after(cache_for, cache)

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
        cache_pass.rewrite_module(op)