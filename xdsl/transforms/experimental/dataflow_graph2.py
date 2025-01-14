from dataclasses import dataclass
import csv
import math

from xdsl.dialects import builtin, func, scf, memref
from xdsl.dialects.experimental import dataflow

from xdsl.ir import (
    BlockArgument,
    MLContext,
    Operation,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
import pickle

def find_module(op):
    module = op
    while not isinstance(module, builtin.ModuleOp):
        module = module.parent_op()

    return module

def calculate_op_latency(op : Operation, mlir_op_latency : dict[str,float]):
    latency = 0
    for body_op in op.walk(): 
        if body_op.results:
            body_op_type = str(body_op.results[0].type)
            encoded_body_op_name = body_op.name + ":" + body_op_type

            if encoded_body_op_name in mlir_op_latency:
                latency += mlir_op_latency[encoded_body_op_name]

    return latency

def get_loop_iters(for_op : scf.For):
    lb = for_op.lb.owner.value.value.data
    ub = for_op.ub.owner.value.value.data
    step = for_op.step.owner.value.value.data
    iters = int((ub-lb)/step + (ub-lb)%step)

    return iters

# TODO: Generates a dictionary with a weight for each relevant MLIR operation. For now, the metric 
# is the latency estimated using the Eucalyptus characterisation, but this is modular to support 
# other metrics such as arithmetic intensity
def get_operation_latencies():
    #csvfile = open("/Users/rodr306/OneDrive - PNNL/Internship/node_balancing/mlir_latencies.txt")
    csvfile = open("/home/rodr306/SC24/end-to-end-example/mlir_latencies.txt")
    latencyreader = csv.reader(csvfile, delimiter=',', quotechar='|')

    mlir_op_latency = dict()

    for row in latencyreader:
        if latencyreader.line_num > 1:
            mlir_op,op_type,latency = row
            mlir_op_latency[mlir_op + ":" + op_type] = float(latency)

    csvfile.close()

    return mlir_op_latency

@dataclass
class NoHierarchyTagFunctionsWithLatencies(RewritePattern):
    function_latency: dict[str, float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        if "node_func" in func_op.attributes:
            mlir_op_latency = get_operation_latencies()

            # Calculate the contribution of each loop to the latency
            for_ops = [body_op for body_op in func_op.walk() if isinstance(body_op, scf.For)]

            func_latency = 0
            for for_op in for_ops:
                # TODO: Generalise this. We are assuming the bounds are arith.constant
                iters = get_loop_iters(for_op)

                for_latency = calculate_op_latency(for_op, mlir_op_latency)

                for_latency *= iters
                func_latency += for_latency

            # Calculate the contribution of the rest of the operations:
            rest_latency = calculate_op_latency(func_op, mlir_op_latency)
            func_latency += rest_latency

            func_op.attributes["latency"] = builtin.FloatAttr(float(func_latency), builtin.Float32Type())
            self.function_latency[func_op.sym_name.data] = func_latency

def is_leaf(node : Operation):
    for node_op in node.walk():
        if isinstance(node_op, func.Call):
            return False
    return True

def get_op_calls(node : func.FuncOp | scf.For):
    calls = []
    for body_op in node.body.ops:
        if isinstance(body_op, func.Call):
            calls.append(body_op)
        elif isinstance(body_op, scf.For):
            calls += get_op_calls(body_op)

    return calls

def discover_children(root : func.FuncOp, function_latency):
    node_calls = get_op_calls(root)
    children_nodes = [builtin.SymbolTable.lookup_symbol(root, node_call.callee) for node_call in node_calls]


    root_latency = root.attributes['latency'].value.data
    children_latency = 0
    for child_node in children_nodes:
        if is_leaf(child_node): # backtrack
            children_latency += child_node.attributes['latency'].value.data
        else:
            children_latency += discover_children(child_node, function_latency)

    # TODO: this currently doesn't work because the top level nodes are encapsulating the first loop nodes. Instead, 
    # the first level nodes should have a loop.
    #if not is_leaf(root):
    #    for_loop = [body_op for body_op in root.body.ops if isinstance(body_op, scf.For)][0]
    for_loop = [body_op for body_op in root.body.ops if isinstance(body_op, scf.For)]
    if for_loop:
        iters = get_loop_iters(for_loop[0])
        root_latency += iters * children_latency
    else:
        root_latency += children_latency

    root.attributes['latency'] = builtin.FloatAttr(root_latency, builtin.Float32Type())
    function_latency[root.sym_name.data] = root_latency
    return root_latency

@dataclass
class HierarchyTagFunctionsWithLatencies(RewritePattern):
    function_latency: dict[str, float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data == "top":
            # We are interested in the first level function calls here, not nested, so that 
            # we can find the leaf nodes that hang from this root and then calculate the latencies 
            # bottom up.
            root = op

            root.attributes['latency'] = builtin.FloatAttr(0.0, builtin.Float32Type())
            root_latency = discover_children(root, self.function_latency)
            root.attributes['latency'] = builtin.FloatAttr(root_latency, builtin.Float32Type())

# TODO: this will be a parameter of the DSE
CHUNK_FACTOR = 2

base_chunk_name_counter = dict()

def chunk_node(top_level_node: dataflow.Node, rewriter: PatternRewriter, operations_to_remove: set[Operation]):
    ## Chunk the loop node by a fixed chunk factor (this will be a parameter of the DSE in the future)
    ## Nodes are deleted after partitioning so only the split versions remain in the IR. We keep track of them in the
    ## operations_to_remove set.

    #for top_level_node in top_level_nodes:
    assert isinstance(top_level_node, func.FuncOp)
    calls = [top_level_op for top_level_op in top_level_node.walk() if isinstance(top_level_op, func.Call)]

    # Use to remove these nodes once they have already been split
    for call in calls:
        called_node = builtin.SymbolTable.lookup_symbol(top_level_node, call.callee.root_reference)
        operations_to_remove.add(called_node)

        # The chunking happens effectively in the for loop of the called node. This requires recalculating the bounds for each
        # clone and TODO: the view of the data for each new loop node
        for_called_node = [called_node_op for called_node_op in called_node.walk() if isinstance(called_node_op, scf.For)]
        if for_called_node:
            for_called_node = for_called_node[0]

            iters = get_loop_iters(for_called_node)
            #n_chunks = int(iters / CHUNK_FACTOR + math.ceil(iters % CHUNK_FACTOR/CHUNK_FACTOR))
            n_chunks = CHUNK_FACTOR

            chunks = []
            chunk_calls = []
            for i in range(n_chunks):
                chunks.append(called_node.clone())
                base_chunk_name = chunks[i].sym_name.data
                if base_chunk_name not in base_chunk_name_counter:
                    base_chunk_name_counter[base_chunk_name] = 0
                chunk_name = base_chunk_name + f"_{str(base_chunk_name_counter[base_chunk_name])}"
                base_chunk_name_counter[base_chunk_name] += 1
                chunks[i].sym_name = builtin.StringAttr(chunk_name)
                chunk_calls.append(func.Call(chunk_name, call.arguments, call.res))
                chunks[i].original_name = base_chunk_name


            lb = for_called_node.lb.owner.value.value.data
            ub = for_called_node.ub.owner.value.value.data
            it_range = ub - lb

            for i in range(n_chunks):
                chunk_for_loop = [chunk_op for chunk_op in chunks[i].walk() if isinstance(chunk_op, scf.For)][0]

                # TODO: adapt this for the case where the number of iterations doesn't divide evenly by the CHUNK_FACTOR 
                chunk_lb = chunk_for_loop.lb.owner.value.value.data
                chunk_size = it_range / CHUNK_FACTOR
                chunk_lb += int(i * chunk_size)
                chunk_ub = int(chunk_lb + chunk_size)

                chunk_for_loop.lb.owner.value = builtin.IntegerAttr.from_index_int_value(chunk_lb)
                chunk_for_loop.ub.owner.value = builtin.IntegerAttr.from_index_int_value(chunk_ub)


            # Data partitioning
            n_chunks = len(chunk_calls)
            for chunk_idx,chunk_call in enumerate(chunk_calls):
                node_inputs = []

                for idx,arg in enumerate(chunk_call.arguments):
                    if isinstance(arg.type, memref.MemRefType):
                        memref_dims = [dim.value.data for dim in arg.type.shape.data]
                        sizes = memref_dims
                        sizes[0] = int(sizes[0] / n_chunks)


                        subview_exists = False
                        for use in arg.uses:
                            op_using_subview = use.operation
                            if isinstance(op_using_subview, memref.Subview) and top_level_node.is_ancestor(op_using_subview):
                                subview = op_using_subview
                                subview_exists = True
                                break
                        if not subview_exists:
                            subview = memref.Subview.from_static_parameters(arg, arg.type, [chunk_idx * sizes[0]], sizes, [1])
                            rewriter.insert_op_before(subview, call)

                        chunk_call.operands[idx] = subview.result

                        node_inputs.append(subview.result.type)
                    else:
                        node_inputs.append(arg)

                # Update the type of the partitioned arguments in the node
                update_function_type = builtin.FunctionType.from_lists(node_inputs, [])
                chunks[chunk_idx].function_type = update_function_type
                for chunk_arg_idx,chunk_arg in enumerate(chunks[chunk_idx].body.block.args):
                    chunk_arg.type = update_function_type.inputs.data[chunk_arg_idx]

            for chunk in chunks:
                rewriter.insert_op_before(chunk, called_node)

            for chunk_call in chunk_calls:
                rewriter.insert_op_before(chunk_call, call)

            rewriter.erase_op(call)

            # Add inter-chunk dependencies:
            for chunk_idx,chunk in enumerate(chunks):
                if chunk_idx > 0: # input token from previous chunk
                    previous_chunk_out_token = chunk_calls[chunk_idx-1].results[0]
                    new_chunk_call_args = list(chunk_calls[chunk_idx].arguments) + [previous_chunk_out_token]
                    chunk_call_with_dependency = func.Call(chunk_calls[chunk_idx].callee, new_chunk_call_args, dataflow.Token())
                    rewriter.insert_op_before(chunk_call_with_dependency, chunk_calls[chunk_idx])
                    rewriter.erase_op(chunk_calls[chunk_idx])
                    chunk_calls[chunk_idx] = chunk_call_with_dependency

                token_out = dataflow.Token()
                chunk_function_type_outputs_list = list(chunk.function_type.outputs.data)
                chunk_function_type_outputs_list.append(token_out)


                function_type_with_token = builtin.FunctionType.from_lists(chunk.function_type.inputs.data, chunk_function_type_outputs_list)
                chunk.function_type = function_type_with_token
                chunk_return = [chunk_op for chunk_op in chunk.body.ops if isinstance(chunk_op, func.Return)][0]

                gen_chunk_token = dataflow.GenToken()
                rewriter.insert_op_before(gen_chunk_token, chunk_return)
                chunk_return.operands = [gen_chunk_token.res]

                # Add the output token to the function call
                chunk_calls[chunk_idx].result_types = [builtin.IntegerAttr.from_int_and_width(2134,32)]
                chunk_call_with_out_token = func.Call(chunk_calls[chunk_idx].callee, chunk_calls[chunk_idx].arguments, dataflow.Token())
                rewriter.insert_op_before(chunk_call_with_out_token, chunk_calls[chunk_idx])
                rewriter.erase_op(chunk_calls[chunk_idx])
                chunk_calls[chunk_idx] = chunk_call_with_out_token # the chunk_calls list is used to create the dependencies

            # This is assuming all the chunk nodes run in parallel, i.e. there were enough resources to instantiate one module per node
            new_top_level_node_latency = top_level_node.attributes['latency'].value.data / CHUNK_FACTOR
            top_level_node.attributes['latency'] = builtin.FloatAttr(new_top_level_node_latency, builtin.Float32Type())

            chunk_for_loop = [chunk_op for chunk_op in chunks[0].walk() if isinstance(chunk_op, scf.For)][0]
            if [chunk_for_op for chunk_for_op in chunk_for_loop.body.walk() if isinstance(chunk_for_op, func.Call)]: # the chunk node has a nested for loop
                for chunk in chunks:
                    chunk_node(chunk, rewriter, operations_to_remove)

@dataclass
class PartitionTopLevelNodes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data == "top":
            top_level_nodes = [builtin.SymbolTable.lookup_symbol(op, body_op.callee.root_reference) for body_op in op.body.ops if isinstance(body_op, func.Call)]

            # Chunk the loop node by a fixed chunk factor (this will be a parameter of the DSE in the future)
            for top_level_node in top_level_nodes:
                chunk_node(top_level_node, rewriter)

@dataclass
class DSE(RewritePattern):
    function_latency: dict[str, float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data == "top":
            graph = pickle.load(open("graph.out", "rb"))
            flow_graph = pickle.load(open("flow.out", "rb"))
            parallel_paths = pickle.load(open("paths.out", "rb"))
            module = find_module(op)
            node_functions_lst = [mod_op for mod_op in module.walk() if isinstance(mod_op, func.FuncOp) and mod_op.sym_name.data != "top"]
            node_functions_dict = dict()
            for node_function in node_functions_lst:
                node_functions_dict[node_function.sym_name.data] = node_function

            # Map path in the flow network u_in -> u_out to path in the dataflow graph with simple nodes
            mapped_parallel_paths = []
            for path in parallel_paths:
                mapped_path = [node for node in path if "_in" not in node.name.root_reference.data]
                for node in mapped_path:
                    node_name = node.name.root_reference.data
                    node.name = builtin.SymbolRefAttr(builtin.StringAttr(node_name.split("_out")[0]))
                mapped_parallel_paths.append(mapped_path)


            # Find the most expensive paralell path
            path_latency_lst = []
            max_path_latency_idx = -1
            max_path_latency = 0

            for path_idx,path in enumerate(mapped_parallel_paths):
                path_latency = 0
                #print("-----> PARALLEL PATH")
                for node in path:
                    #print("NODE: ", node.name)
                    path_latency += self.function_latency[node.name.root_reference.data]

                #print("PATH LATENCY: ", path_latency, "\n")
                if path_latency > max_path_latency:
                    max_path_latency = path_latency
                    max_path_latency_idx = path_idx

                path_latency_lst.append(path_latency)

            # Find the most expensive node in the most expensive parallel path
            most_expensive_path = mapped_parallel_paths[max_path_latency_idx]
            most_expensive_node = None
            highest_path_latency = 0

            for node in most_expensive_path:
                node_latency = node_functions_dict[node.name.root_reference.data].attributes["latency"].value.data

                if node_latency > highest_path_latency:
                    most_expensive_node = node
                    highest_path_latency = node_latency

            #print("MOST EXPENSIVE NODE: ", most_expensive_node.name)

            most_expensive_node_function = [node_function for node_function in find_module(op).ops if isinstance(node_function, func.FuncOp) and node_function.sym_name == most_expensive_node.name.root_reference][0]
            #most_expensive_node_function = [node_function.sym_name for node_function in find_module(op).ops if isinstance(node_function, func.FuncOp)]
            #print(most_expensive_node_function)
            operations_to_remove : set[Operation] = set()
            chunk_node(most_expensive_node_function, rewriter, operations_to_remove)

            #for operation_to_remove in operations_to_remove:
            #    operation_to_remove.detach()
            #    operation_to_remove.erase()

@dataclass
class DataflowGraph2(ModulePass):
    name = "dataflow-graph2"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        function_latency = dict()

        noh_latency_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    NoHierarchyTagFunctionsWithLatencies(function_latency),
                ]
            ),
            apply_recursively=False,
        )
        noh_latency_pass.rewrite_module(op)

        h_latency_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    HierarchyTagFunctionsWithLatencies(function_latency)
                ]
            ),
            apply_recursively=False,
        )
        h_latency_pass.rewrite_module(op)

        partition_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PartitionTopLevelNodes()
                ]
            ),
            apply_recursively=False,
        )
        #partition_pass.rewrite_module(op)

        dse_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DSE(function_latency)
                ]
            ),
            apply_recursively=False,
        )
        dse_pass.rewrite_module(op)