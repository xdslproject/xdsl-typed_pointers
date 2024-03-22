from dataclasses import dataclass

from xdsl.dialects import builtin, func
from xdsl.dialects.experimental.hls import PragmaDataflow
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

function_list = []
function_names_used = dict()


def is_parent(op, parent):
    current_op = op.owner

    while not isinstance(current_op, builtin.ModuleOp):
        if current_op == parent:
            return True
        current_op = current_op.parent_op()

    return False


ops_out_of_function = dict()


@dataclass
class DataflowToFunc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaDataflow, rewriter: PatternRewriter, /):
        dataflow_body = op.body
        op.detach_region(dataflow_body)
        df_function_name = "df_function"
        if "function" in op.attributes:
            df_function_name = op.attributes["function"].data

        if df_function_name in function_names_used:
            function_names_used[df_function_name] += 1
            df_function_name += "_" + str(function_names_used[df_function_name])
        else:
            function_names_used[df_function_name] = 0
            df_function_name += "_0"
        df_function = func.FuncOp.from_region(df_function_name, [], [], dataflow_body)

        # Check which operands have definitions outside the dataflow function. They should
        # either copied in the body of the function or become arugments.
        ops_out_of_function[df_function] = set()
        for func_op in df_function.body.ops:
            for operand in func_op.operands:
                if not is_parent(operand, df_function):
                    ops_out_of_function[df_function].add(operand)

        input_lst = [_op.type for _op in ops_out_of_function[df_function]]
        df_function.function_type = builtin.FunctionType.from_lists(input_lst, [])
        for _type in reversed(input_lst):
            df_function.body.blocks[0].insert_arg(_type, 0)

        function_list.append(df_function)

        op.detach()
        op.erase()


@dataclass
class EraseModule(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter, /):
        for _op in op.ops:
            _op.detach()
            _op.erase(drop_references=False)

        op.body.blocks[0].add_ops(function_list)


@dataclass
class InjectFunctionsInModule(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter, /):
        op.body.blocks[0].add_ops(function_list)


@dataclass
class SplitDataflowPass(ModulePass):
    name = "split-dataflow"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        dataflow_to_func = PatternRewriteWalker(
            DataflowToFunc(), apply_recursively=False, walk_reverse=False
        )
        dataflow_to_func.rewrite_module(op)

        inject_functions_in_module = PatternRewriteWalker(
            InjectFunctionsInModule(), apply_recursively=False, walk_reverse=False
        )
        inject_functions_in_module.rewrite_module(op)

        erase_module = PatternRewriteWalker(
            EraseModule(), apply_recursively=False, walk_reverse=False
        )
        erase_module.rewrite_module(op)
