from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.dialects.experimental import dataflow
#from xdsl.transforms.experimental.dataflow_to_func import ConvertDataflowToFunc

from xdsl.ir import (
    MLContext,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass
class GenerateGraphFromConnected(RewritePattern):
    events : list[str]
    events_waitlist : list[str]
    depends : dict[str,str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, connected_op: dataflow.Connected, rewriter: PatternRewriter):
        node = connected_op.node

        in_nodes_lst = connected_op.in_nodes.data
        out_nodes_lst = connected_op.out_nodes.data

        if in_nodes_lst:
            events_in_waitlist = ",".join([f"{in_node_symbol.root_reference.data}_done" for in_node_symbol in in_nodes_lst])
            self.events_waitlist[0] += f"cl_event {node.root_reference.data}_waitlist[] = {{{events_in_waitlist}}};\n\t"

            node_name = node.root_reference.data
            self.depends[node_name] = []
            for in_node_symbol in in_nodes_lst:
                self.depends[node_name].append(in_node_symbol.root_reference.data)

        #if out_nodes_lst: # TODO: for now we generate a completion event for every node, even if it doesn't have successors
        self.events[0] += f"cl_event {node.root_reference.data}_done;\n\t"

@dataclass
class GenerateKernelsAndBuffersCode(RewritePattern):
    depends : dict[str,str]
    host_arrays: list[str]
    init_host_arrays: list[str]
    iter_vars: list[str]
    create_kernel: list[str]
    create_buffer: list[str]
    set_kernel_arg: list[str]
    enqueue_kernel: list[str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, node_op: dataflow.Node, rewriter: PatternRewriter):
        node_name = node_op.sym_name.data
        self.host_arrays[0] += f"float * out_{node_name} = (float*)malloc(N * sizeof(float));\n\t"
        self.init_host_arrays[0] += f"for(int i = 0; i < N; i++) out_{node_name}[i] = 0;\n\t"
        self.iter_vars[0] += f"cl_long iters_{node_name} = N;\n\t"
        self.create_kernel[0] += f"cl_kernel {node_name}_kernel = clCreateKernel(program, \"{node_name}\", &err);\n\t"
        self.create_buffer[0] += f"cl_mem out_{node_name}_buf = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), out_{node_name}, NULL);\n\t"
        if node_name == "node_0":
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_0_kernel, 0, sizeof(cl_mem), &in_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_0_kernel, 1, sizeof(cl_mem), &out_node_0_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_0_kernel, 2, sizeof(cl_long), &iters_{node_name});\n\t"
        elif node_name == "node_4": # TODO: this works for the example with 5 nodes, but we should count the number of nodes to make this generic
            for n_pred,pred_node in enumerate(self.depends[node_name]):
                self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_4_kernel, {n_pred}, sizeof(cl_mem), &out_{pred_node}_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_4_kernel, {n_pred+1}, sizeof(cl_mem), &out_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_4_kernel, {n_pred+2}, sizeof(cl_long), &iters_{node_name});\n\t"
        else:
            for n_pred,pred_node in enumerate(self.depends[node_name]):
                self.set_kernel_arg[0] += f"err  = clSetKernelArg({node_name}_kernel, {n_pred}, sizeof(cl_mem), &out_{pred_node}_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg(node_4_kernel, {n_pred+1}, sizeof(cl_mem), &out_{node_name});\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg({node_name}_kernel, {n_pred+2}, sizeof(cl_long), &iters_{node_name});\n\t"

        if node_name in self.depends:
            self.enqueue_kernel[0] += f"err = clEnqueueTask(commands, {node_name}_kernel, {len(self.depends[node_name])}, {node_name}_waitlist, &{node_name}_done);\n\t"
        else:
            self.enqueue_kernel[0] += f"err = clEnqueueTask(commands, {node_name}_kernel, 0, NULL, &{node_name}_done);\n\t"


def print_boilerplate(host_arrays, init_host_arrays, iter_vars, create_kernel, create_buffer, set_kernel_arg, events, events_waitlist, enqueue_kernel):
    boilerplate = f"""#include <stdio.h>\n
    #include <stdlib.h>\n
    #include <CL/cl.h>\n
                      \n
    #define N 100     \n
                      \n
    int main() {{      \n
        cl_device_id device_id;\n
        int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);\n
        cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);\n
        if (!context) \n
        {{             \n
            printf(\"Error: Failed to create a compute context!\\n\");\n
            return EXIT_FAILURE;\n
        }}             \n
                      \n
        // Create a command commands\n
        //            \n
        cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);\n
        if (!commands)\n
        {{             \n
            printf(\"Error: Failed to create a command commands!\\n\");\n
            return EXIT_FAILURE;\n
        }}             \n
                      \n
                      \n
                      \n
        FILE * f;     \n
        f = fopen(\"all_nodes.xclbin\", \"r\");\n
                      \n
        fseek(f, 0, SEEK_END);\n
        size_t file_size = ftell(f);\n
        fseek(f, 0, SEEK_SET);\n
                      \n
        const unsigned char * binary = malloc(file_size * sizeof(const unsigned char));\n
        fread(binary, file_size, 1, f);\n
                      \n
                      \n
        cl_int binary_status;\n
        cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &file_size, &binary, &binary_status, &err);\n
        if (!program) {{\n
            printf(\"Error: Failed to create compute program!\\n\");\n
        }}             \n
                      \n
                      \n
        //// Build the program executable\n
        //            \n
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);\n
        if (err != CL_SUCCESS)\n
        {{             \n
            size_t len;\n
            char buffer[2048];\n
                      \n
            printf(\"Error: Failed to build program executable!\\n\");\n
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);\n
            printf(\"%s\\n\", buffer);\n
            exit(1);  \n
        }}             \n
        {iter_vars}
                      \n
        float * in = (float *)malloc(N * sizeof(float));\n
        float * out = (float *)malloc(N * sizeof(float));\n
        {host_arrays}
                      \n
        for(int i = 0; i < N; i++) {{\n
            in[i] = i+1;\n
            out[i] = 0;\n
        }}             \n
        {init_host_arrays}
                      \n
        {create_kernel}
                      \n
        cl_mem in_buf = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), in, NULL);\n\t
        cl_mem out_buf = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), out, NULL);\n\t
        {create_buffer}
                      \n
        {set_kernel_arg}
                      \n
        {events}
                      \n
        {events_waitlist}\n
                      \n
        {enqueue_kernel}\n
                      \n
        clFinish(commands);\n
                      \n
        err = clEnqueueReadBuffer( commands, out_buf, CL_TRUE, 0, N * sizeof(float), out, 0, NULL, NULL );      \n
                      \n
        for(int i = 0; i < N; i++) {{\n
            printf(\"out[%d] = %f\\n\", i, out[i]);\n
            out[i] = 0;\n
        }}             \n
    }}                 \n"""
    print(boilerplate)


@dataclass
class PrintHostCodeDataflowVitis(ModulePass):
    name = "print-host-code-dataflow-vitis"


    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        events = [""]
        events_waitlists = [""]
        depends = dict()

        host_arrays = [""]
        init_host_arrays = [""]
        iter_vars = [""]
        create_kernel = [""]
        create_buffer = [""]
        set_kernel_arg = [""]
        enqueue_kernel = [""]

        #print_boilerplate()

        generate_graph_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GenerateGraphFromConnected(events, events_waitlists, depends),
                ]
            ),
            apply_recursively=False,
        )
        generate_graph_pass.rewrite_module(op)

        generate_kernels_buffer_code_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GenerateKernelsAndBuffersCode(depends, host_arrays, init_host_arrays, iter_vars, create_kernel, create_buffer, set_kernel_arg, enqueue_kernel),
                ]
            ),
            apply_recursively=False,
        )
        generate_kernels_buffer_code_pass.rewrite_module(op)

        print_boilerplate(host_arrays[0], init_host_arrays[0], iter_vars[0], create_kernel[0], create_buffer[0], set_kernel_arg[0], events[0], events_waitlists[0], enqueue_kernel[0])