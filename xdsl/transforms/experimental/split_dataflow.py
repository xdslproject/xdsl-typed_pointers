from dataclasses import dataclass

from xdsl.dialects import builtin, func, scf
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


def propagate_for_args(op, df_function):
    for operand in op.operands:
        if not is_parent(operand, df_function):
            operand.owner.detach()
            df_function.body.blocks[0].insert_op_before(
                operand.owner, df_function.body.blocks[0].first_op
            )
            propagate_for_args(operand.owner, df_function)


def walk_body(body, input_lst, df_function):
    for op in body.ops:
        if isinstance(op, scf.For) or isinstance(op, scf.ParallelOp):
            propagate_for_args(op, df_function)

        for other_body in op.regions:
            walk_body(other_body, input_lst, df_function)

        if not (isinstance(op, scf.For) or isinstance(op, scf.ParallelOp)):
            operand_idx = 0
            for operand in op.operands:
                if not is_parent(operand, df_function):
                    n_args = len(df_function.body.blocks[0].args)
                    df_function.body.blocks[0].insert_arg(operand.type, n_args)
                    op.operands[operand_idx] = df_function.body.blocks[0].args[n_args]
                    input_lst.append(operand.type)
                operand_idx += 1


def walk_function_ops(df_function):
    input_lst = []

    walk_body(df_function.body, input_lst, df_function)

    df_function.function_type = builtin.FunctionType.from_lists(input_lst, [])


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
        # either be copied in the body of the function or become arguments.
        #
        # Here the operands whose parent operations are not in the dataflow function are added
        # as arguments to the function. The corresponding arguments are added to the block and
        # linked to each operand.
        walk_function_ops(df_function)

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
