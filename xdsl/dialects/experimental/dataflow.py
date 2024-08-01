from __future__ import annotations

from xdsl.ir import Dialect, Region, Attribute, Sequence, Block, SSAValue, Operation
from xdsl.irdl import irdl_op_definition, region_def, IRDLOperation, VarOperand, AnyAttr, var_result_def, var_operand_def, VarOpResult, opt_attr_def, operand_def, irdl_attr_definition, ParametrizedAttribute, ParameterDef, attr_def
from xdsl.dialects.builtin import StringAttr, FunctionType, SymbolRefAttr, ArrayAttr, DictionaryAttr, IndexType
from xdsl.dialects import builtin, memref, func
from xdsl.traits import HasParent, IsTerminator, SymbolOpInterface, IsolatedFromAbove
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.rewriter import Rewriter
from xdsl.builder import Builder
from typing import cast

from xdsl.printer import Printer
from xdsl.parser import Parser

from xdsl.dialects.utils import print_func_op_like, print_return_op_like, parse_call_op_like, print_call_op_like, parse_func_op_like

from xdsl.utils.hints import isa


@irdl_op_definition
class Top(IRDLOperation):
    name = "df.top"

    body: Region = region_def()
    sym_name: StringAttr = attr_def(StringAttr)
    function_type: FunctionType = attr_def(FunctionType)
    sym_visibility: StringAttr | None = opt_attr_def(StringAttr)
    arg_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_attr_def(ArrayAttr[DictionaryAttr])

    traits = frozenset(
        [IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface()]
    )

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
        visibility: StringAttr | str | None = None,
        *,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
    ):
        if isinstance(visibility, str):
            visibility = StringAttr(visibility)
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
            "sym_visibility": visibility,
            "arg_attrs": arg_attrs,
            "res_attrs": res_attrs,
        }
        super().__init__(attributes=attributes, regions=[region])

    @staticmethod
    def from_function(function : func.FuncOp) -> Top:
        return_op = function.get_return_op()
        if return_op:
            Rewriter.replace_op(return_op, TopEnd())

        top_body = Rewriter.move_region_contents_to_new_regions(function.body)
        
        return Top(name="top", function_type=function.function_type, region=top_body)
    
    @classmethod
    def parse(cls, parser: Parser) -> Top:
        visibility = parser.parse_optional_visibility_keyword()

        (
            name,
            input_types,
            return_types,
            region,
            extra_attrs,
            arg_attrs,
        ) = parse_func_op_like(
            parser, reserved_attr_names=("sym_name", "function_type", "sym_visibility")
        )
        func = Top(
            name=name,
            function_type=(input_types, return_types),
            region=region,
            visibility=visibility,
            arg_attrs=arg_attrs,
        )
        if extra_attrs is not None:
            func.attributes |= extra_attrs.data
        return func

    #@classmethod
    #def parse(cls, parser: Parser) -> Top:
    #    # Parse visibility keyword if present
    #    if parser.parse_optional_keyword("public"):
    #        visibility = "public"
    #    elif parser.parse_optional_keyword("nested"):
    #        visibility = "nested"
    #    elif parser.parse_optional_keyword("private"):
    #        visibility = "private"
    #    else:
    #        visibility = None

    #    # Parse function name
    #    name = parser.parse_symbol_name().data

    #    def parse_fun_input():
    #        ret = parser.parse_optional_argument()
    #        if ret is None:
    #            ret = parser.parse_optional_type()
    #        if ret is None:
    #            parser.raise_error("Expected argument or type")
    #        return ret

    #    # Parse function arguments
    #    args = parser.parse_comma_separated_list(
    #        parser.Delimiter.PAREN,
    #        parse_fun_input,
    #    )

    #    # Check consistency (They should be either all named or none)
    #    if isa(args, list[parser.Argument]):
    #        entry_args = args
    #        input_types = cast(list[Attribute], [a.type for a in args])
    #    elif isa(args, list[Attribute]):
    #        entry_args = None
    #        input_types = args
    #    else:
    #        parser.raise_error(
    #            "Expected all arguments to be named or all arguments to be unnamed."
    #        )

    #    # Parse return type
    #    if parser.parse_optional_punctuation("->"):
    #        return_types = parser.parse_optional_comma_separated_list(
    #            parser.Delimiter.PAREN, parser.parse_type
    #        )
    #        if return_types is None:
    #            return_types = [parser.parse_type()]
    #    else:
    #        return_types = []

    #    attr_dict = parser.parse_optional_attr_dict_with_keyword(
    #        ("sym_name", "function_type", "sym_visibility")
    #    )

    #    # Parse body
    #    region = parser.parse_optional_region(entry_args)
    #    if region is None:
    #        region = Region()
    #    top_function = func.FuncOp.from_region(name, input_types, return_types, region, visibility)
    #    top = Top.from_function(top_function)
    #    if attr_dict is not None:
    #        func.attributes |= attr_dict.data
    #    return top

    
    def print(self, printer: Printer):
        if self.sym_visibility:
            visibility = self.sym_visibility.data
            printer.print(f" {visibility}")

        print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            arg_attrs=self.arg_attrs,
            reserved_attr_names=(
                "sym_name",
                "function_type",
                "sym_visibility",
                "arg_attrs",
            ),
        )
    #def print(self, printer: Printer):
    #    reserved = {"sym_name", "function_type", "sym_visibility"}
    #    if self.sym_visibility:
    #        visibility = self.sym_visibility.data
    #        printer.print(f" {visibility}")

    #    printer.print(f" @{self.sym_name.data}")
    #    if len(self.body.blocks) > 0:
    #        printer.print("(")
    #        printer.print_list(self.body.blocks[0].args, printer.print_block_argument)
    #        printer.print(") ")
    #        if self.function_type.outputs:
    #            printer.print("-> ")
    #            if len(self.function_type.outputs) > 1:
    #                printer.print("(")
    #            printer.print_list(self.function_type.outputs, printer.print_attribute)
    #            if len(self.function_type.outputs) > 1:
    #                printer.print(")")
    #            printer.print(" ")
    #    else:
    #        printer.print_attribute(self.function_type)
    #    printer.print_op_attributes_with_keyword(self.attributes, reserved)

    #    if len(self.body.blocks) > 0:
    #        printer.print_region(self.body, False, False)
    
@irdl_op_definition
class TopEnd(IRDLOperation):
    name = "df.top_end"

    traits = frozenset([HasParent(Top), IsTerminator()])

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, [])

    #def print(self, printer: Printer):
    #    if self.attributes:
    #        printer.print(" ")
    #        printer.print_op_attributes(self.attributes)

    #    if self.arguments:
    #        printer.print(" ")
    #        printer.print_list(self.arguments, printer.print_ssa_value)
    #        printer.print(" : ")
    #        printer.print_list(
    #            (x.type for x in self.arguments), printer.print_attribute
    #        )

@irdl_op_definition
class Node(IRDLOperation):
    name = "df.node"

    body: Region = region_def()
    sym_name: StringAttr = attr_def(StringAttr)
    node_type: FunctionType = attr_def(FunctionType)

    traits = frozenset(
        [SymbolOpInterface()]
    )

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        attributes: dict[str, Attribute | None] = {
            "sym_name": StringAttr(name),
            "node_type": function_type,
        }
        super().__init__(attributes=attributes, regions=[region])

    def print(self, printer: Printer):
        print_func_op_like(
            printer,
            self.sym_name,
            self.node_type,
            self.body,
            self.attributes,
            reserved_attr_names=(
                "sym_name",
                "node_type",
            ),
        )

@irdl_op_definition
class NodeEnd(IRDLOperation):
    name = "df.node_end"

    traits = frozenset([HasParent(Node), IsTerminator()])

    def print(self, printer: Printer):
        print_return_op_like(printer, self.attributes, [])

@irdl_op_definition
class NodeCall(IRDLOperation):
    name = "df.node_call"
    arguments: VarOperand = var_operand_def(AnyAttr())
    callee: SymbolRefAttr = attr_def(SymbolRefAttr)
    args_directions = opt_attr_def(ArrayAttr[StringAttr])

    # Note: naming this results triggers an ArgumentError
    res: VarOpResult = var_result_def(AnyAttr())

    # TODO how do we verify that the types are correct?
    def __init__(
        self,
        callee: str | SymbolRefAttr,
        arguments: Sequence[SSAValue | Operation],
        return_types: Sequence[Attribute],
        args_directions: ArrayAttr[StringAttr]
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[arguments],
            result_types=[return_types],
            attributes={"callee": callee, "args_directions": args_directions}
        )

    def print(self, printer: Printer):
        print_call_op_like(
            printer,
            self,
            self.callee,
            self.arguments,
            self.attributes,
            reserved_attr_names=("callee",),
        )

    @classmethod
    def parse(cls, parser: Parser) -> NodeCall:
        callee, arguments, results, extra_attributes = parse_call_op_like(
            parser, reserved_attr_names=("callee",)
        )
        call = NodeCall(callee, arguments, results)
        if extra_attributes is not None:
            call.attributes |= extra_attributes.data
        return call
    
def process_memref_alloc(memalloc : memref.Alloc, arc_neighbours : list, node_call : NodeCall):
    for alloc_use in memalloc.memref.uses:
        if isinstance(alloc_use.operation, NodeCall) and alloc_use.operation != node_call:
            neighbour_node_call = alloc_use.operation
            neighbour_direction = neighbour_node_call.args_directions.data[alloc_use.index].data
            arc_neighbours.append([alloc_use.operation, neighbour_direction])

@irdl_op_definition
class Connected(IRDLOperation):
    name = "df.connected"

    node = attr_def(SymbolRefAttr)
    in_nodes = attr_def(ArrayAttr[SymbolRefAttr])
    out_nodes = attr_def(ArrayAttr[SymbolRefAttr])
    edges = attr_def(DictionaryAttr)
    arguments: VarOperand = var_operand_def(AnyAttr())

    def __init__(
            self,
            node_call : NodeCall
    ):  
        in_nodes = []
        out_nodes = []
        edges = dict()

        node_sym = node_call.callee

        for operand_idx, operand in enumerate(node_call.operands):
            arc_neighbours = []

            if isinstance(operand.owner, memref.AlterShapeOp):
                altershape = operand.owner
                if isinstance(altershape.src.op, memref.Alloc):
                    process_memref_alloc(altershape.src.op, arc_neighbours, node_call)

            if isinstance(operand.owner, memref.Alloc):
                process_memref_alloc(operand.owner, arc_neighbours, node_call)
            else:
                for use in operand.uses:
                    if isinstance(use.operation, NodeCall) and use.operation != node_call:
                        user_node_call = use.operation
                        neighbour_direction = user_node_call.args_directions.data[use.index].data
                        arc_neighbours.append([user_node_call, neighbour_direction])

            node_direction = node_call.args_directions.data[operand_idx].data

            for arc_list in arc_neighbours:
                arc_neighbour = arc_list[0]
                other_direction = arc_list[1]

                # direction has the perspective of the neighbour node, hence it is reversed for 
                # the current node.
                is_arc = False

                if other_direction == "in" and (node_direction == "out" or node_direction == "inout"):
                    out_nodes.append(arc_neighbour.callee)
                    is_arc = True
                elif other_direction == "out" and (node_direction == "in" or node_direction == "inout"):
                    in_nodes.append(arc_neighbour.callee)
                    is_arc = True
                elif node_direction == "inout":
                    in_nodes.append(arc_neighbour.callee)
                    out_nodes.append(arc_neighbour.callee)
                    is_arc = True

                if is_arc:
                    if operand_idx not in edges:
                        edges[operand_idx] = [ArrayAttr([arc_neighbour.callee, operand.type])]
                    else:
                        edges[operand_idx].append(ArrayAttr([arc_neighbour.callee, operand.type]))

                    # Sorting the edges to produce the same result in every run. This is necessary as the use field is a set
                    edges[operand_idx] = sorted(edges[operand_idx], key=lambda x: x.data[0].root_reference.data)

        # Same as above
        in_nodes = ArrayAttr(sorted(in_nodes, key=lambda x: x.root_reference.data))
        out_nodes = ArrayAttr(sorted(out_nodes, key=lambda x: x.root_reference.data))

        for operand in edges.keys():
            edges[operand] = ArrayAttr(edges[operand])
        edges = DictionaryAttr(edges)

        super().__init__(operands=[node_call.arguments], attributes={"node": node_sym, "in_nodes": in_nodes, "out_nodes": out_nodes, "edges": edges})



Dataflow = Dialect(
    [
        Node,
        NodeEnd
    ],
    []
)