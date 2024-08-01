from dataclasses import dataclass

from xdsl.dialects import builtin, affine, func, arith, memref, linalg
from xdsl.dialects.experimental import dataflow

from xdsl import traits
from xdsl.ir import (
    BlockArgument,
    MLContext,
    Operation,
    OpResult
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