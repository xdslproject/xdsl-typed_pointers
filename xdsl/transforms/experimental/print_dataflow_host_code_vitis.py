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
    events_waitlists: dict[str,str]
    depends : dict[str,str]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, connected_op: dataflow.Connected, rewriter: PatternRewriter):
        node = connected_op.node
        node_name = node.root_reference.data

        in_nodes_lst = connected_op.in_nodes.data
        out_nodes_lst = connected_op.out_nodes.data

        if in_nodes_lst:
            events_in_waitlist = ",".join([f"{in_node_symbol.root_reference.data}_done" for in_node_symbol in in_nodes_lst])
            self.events_waitlists[node_name] = f"cl_event {node.root_reference.data}_waitlist[] = {{{events_in_waitlist}}};\n\t"

            node_name = node.root_reference.data
            self.depends[node_name] = []
            for in_node_symbol in in_nodes_lst:
                self.depends[node_name].append(in_node_symbol.root_reference.data)

        #if out_nodes_lst: # TODO: for now we generate a completion event for every node, even if it doesn't have successors
        self.events[0] += f"cl_event {node.root_reference.data}_done;\n\t"

@dataclass
class GenerateKernelsAndBuffersCode(RewritePattern):
    depends : dict[str,str]
    events_waitlists : dict[str,str]
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
            self.set_kernel_arg[0] += f"err  = clSetKernelArg({node_name}_kernel, {n_pred+1}, sizeof(cl_mem), &out_{node_name}_buf);\n\t"
            self.set_kernel_arg[0] += f"err  = clSetKernelArg({node_name}_kernel, {n_pred+2}, sizeof(cl_long), &iters_{node_name});\n\t"

        if node_name in self.depends:
            self.enqueue_kernel[0] += self.events_waitlists[node_name]
            self.enqueue_kernel[0] += f"err = clEnqueueTask(commands, {node_name}_kernel, {len(self.depends[node_name])}, {node_name}_waitlist, &{node_name}_done);\n\t"
        else:
            self.enqueue_kernel[0] += f"err = clEnqueueTask(commands, {node_name}_kernel, 0, NULL, &{node_name}_done);\n\t"


def print_boilerplate(host_arrays, init_host_arrays, iter_vars, create_kernel, create_buffer, set_kernel_arg, events, enqueue_kernel):
    boilerplate = f"""
    #include <stdio.h>
    #include <stdlib.h>
    #include <CL/cl.h>
    #include "omp.h"
                      
    #define N 100     
                      
    int main() {{      
        cl_device_id device_id;
        int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, NULL);
        cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if (!context) 
        {{             
            printf(\"Error: Failed to create a compute context!\\n");
            return EXIT_FAILURE;
        }}             
                      
        // Create a command commands
        //            
        cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
        if (!commands)
        {{             
            printf(\"Error: Failed to create a command commands!\\n");
            return EXIT_FAILURE;
        }}             
                      
                      
                      
        FILE * f;     
        f = fopen(\"all_nodes.xclbin\", \"r\");
                      
        fseek(f, 0, SEEK_END);
        size_t file_size = ftell(f);
        fseek(f, 0, SEEK_SET);
                      
        const unsigned char * binary = malloc(file_size * sizeof(const unsigned char));
        fread(binary, file_size, 1, f);
                      
                      
        cl_int binary_status;
        cl_program program = clCreateProgramWithBinary(context, 1, &device_id, &file_size, &binary, &binary_status, &err);
        if (!program) {{
            printf(\"Error: Failed to create compute program!\\n");
        }}             
                      
                      
        //// Build the program executable
        //            
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS)
        {{             
            size_t len;
            char buffer[2048];
                      
            printf(\"Error: Failed to build program executable!\\n");
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            printf(\"%s\\n", buffer);
            exit(1);  
        }}             
        {iter_vars}
                      
        float * in = (float *)malloc(N * sizeof(float));
        float * out = (float *)malloc(N * sizeof(float));
        {host_arrays}
                      
        for(int i = 0; i < N; i++) {{
            in[i] = i+1;
            out[i] = 0;
        }}             
        {init_host_arrays}
                      
        {create_kernel}
                      
        cl_mem in_buf = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), in, NULL);\t
        cl_mem out_buf = clCreateBuffer(context,  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), out, NULL);\t
        {create_buffer}
                      
        {set_kernel_arg}
                      
        {events}
                      
        double start = omp_get_wtime();
        {enqueue_kernel}
                      
        clFinish(commands);
        double exec_time = omp_get_wtime() - start;
        printf("Execution time: %lf\\n", exec_time); 
                      
        err = clEnqueueReadBuffer( commands, out_buf, CL_TRUE, 0, N * sizeof(float), out, 0, NULL, NULL );      
                      
        for(int i = 0; i < N; i++) {{
            printf(\"out[%d] = %f\\n", i, out[i]);
            out[i] = 0;
        }}             
    }}                 """
    print(boilerplate)


@dataclass
class PrintHostCodeDataflowVitis(ModulePass):
    name = "print-host-code-dataflow-vitis"


    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        events = [""]
        events_waitlists = dict()
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
                    GenerateKernelsAndBuffersCode(depends, events_waitlists, host_arrays, init_host_arrays, iter_vars, create_kernel, create_buffer, set_kernel_arg, enqueue_kernel),
                ]
            ),
            apply_recursively=False,
        )
        generate_kernels_buffer_code_pass.rewrite_module(op)

        print_boilerplate(host_arrays[0], init_host_arrays[0], iter_vars[0], create_kernel[0], create_buffer[0], set_kernel_arg[0], events[0], enqueue_kernel[0])