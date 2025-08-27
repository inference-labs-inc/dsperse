#!/usr/bin/env python3
"""
EZKL-Compatible Conv Decomposer, Splitter, and Metadata Generator (Final Version)

This script automates the entire workflow for preparing models for EZKL:
1. Replaces 'Conv' layers with an FFT-based circuit using only EZKL-compatible
   operators (MatMul, Mul, Pad, etc.), pinning the opset to 18.
2. The kernel's FFT is pre-computed to simplify the in-circuit computation.
3. Splits the result into three separate ONNX files (FFT, Mul, IFFT).
4. Generates a correctly chained metadata.json for all created circuit files.
"""
import os
import json
import shutil
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# --- Helper functions for MatMul-based FFT ---

def get_dft_matrix(n: int, inverse: bool = False):
    """Generates the DFT matrix W for FFT via MatMul, where FFT(x) = Wx."""
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(-2j * np.pi / n)
    w_complex = np.power(omega, -i * j if inverse else i * j)
    # For IFFT, we also need to scale by 1/n. We'll do this in the graph.
    return w_complex.astype(np.complex64)

def precompute_kernel_fft(kernel_data, target_shape):
    """Pre-computes the 2D FFT of the kernel using NumPy."""
    pad_h = target_shape[2] - kernel_data.shape[2]
    pad_w = target_shape[3] - kernel_data.shape[3]
    padded_kernel = np.pad(kernel_data, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    fft_kernel = np.fft.fft2(padded_kernel, axes=(-2, -1))
    return fft_kernel.astype(np.complex64)


class ConvSplitterForEZKL:
    """Decomposes a Conv node and saves the 3 circuit files for EZKL."""

    def __init__(self, model_path: str, output_dir: str, base_name: str):
        self.model_path = model_path
        self.output_dir = output_dir
        self.base_name = base_name
        self.graph = gs.import_onnx(onnx.load(model_path))
        os.makedirs(self.output_dir, exist_ok=True)
        self.created_files = []

    def process(self) -> list:
        conv_node = next((node for node in self.graph.nodes if node.op == "Conv"), None)
        
        if not conv_node:
            print(f"  - No 'Conv' in {os.path.basename(self.model_path)}. Copying.")
            dest_path = os.path.join(self.output_dir, f"{self.base_name}.onnx")
            shutil.copy2(self.model_path, dest_path)
            self.created_files.append(dest_path)
            return self.created_files

        print(f"  - Decomposing 'Conv' in {os.path.basename(self.model_path)} for EZKL...")
        self._decompose_and_split(conv_node)
        return self.created_files
        
    def _create_matmul_fft_subgraph(self, graph, complex_input_tuple, w_matrix, h_matrix):
        """Creates a 2D FFT subgraph using MatMul on real and imaginary parts."""
        real_in, imag_in = complex_input_tuple
        w_r, w_i = gs.Constant("W_real", np.real(w_matrix)), gs.Constant("W_imag", np.imag(w_matrix))
        h_r, h_i = gs.Constant("H_real", np.real(h_matrix)), gs.Constant("H_imag", np.imag(h_matrix))

        def complex_matmul(x_r, x_i, w_r_const, w_i_const, suffix):
            term1 = gs.Variable(f"term1_{suffix}"); node1 = gs.Node(op="MatMul", inputs=[x_r, w_r_const], outputs=[term1])
            term2 = gs.Variable(f"term2_{suffix}"); node2 = gs.Node(op="MatMul", inputs=[x_i, w_i_const], outputs=[term2])
            out_r = gs.Variable(f"out_r_{suffix}"); node3 = gs.Node(op="Sub", inputs=[term1, term2], outputs=[out_r])
            term3 = gs.Variable(f"term3_{suffix}"); node4 = gs.Node(op="MatMul", inputs=[x_r, w_i_const], outputs=[term3])
            term4 = gs.Variable(f"term4_{suffix}"); node5 = gs.Node(op="MatMul", inputs=[x_i, w_r_const], outputs=[term4])
            out_i = gs.Variable(f"out_i_{suffix}"); node6 = gs.Node(op="Add", inputs=[term3, term4], outputs=[out_i])
            return out_r, out_i, [node1, node2, node3, node4, node5, node6]

        # FFT along width
        transposed_r = gs.Variable("transposed_r_w"); graph.nodes.append(gs.Node(op="Transpose", inputs=[real_in], outputs=[transposed_r], attrs={"perm": [0, 1, 3, 2]}))
        transposed_i = gs.Variable("transposed_i_w"); graph.nodes.append(gs.Node(op="Transpose", inputs=[imag_in], outputs=[transposed_i], attrs={"perm": [0, 1, 3, 2]}))
        fft_w_r, fft_w_i, nodes_w = complex_matmul(transposed_r, transposed_i, w_r, w_i, "w")
        graph.nodes.extend(nodes_w)

        # FFT along height
        transposed_r_h = gs.Variable("transposed_r_h"); graph.nodes.append(gs.Node(op="Transpose", inputs=[fft_w_r], outputs=[transposed_r_h], attrs={"perm": [0, 1, 3, 2]}))
        transposed_i_h = gs.Variable("transposed_i_h"); graph.nodes.append(gs.Node(op="Transpose", inputs=[fft_w_i], outputs=[transposed_i_h], attrs={"perm": [0, 1, 3, 2]}))
        fft_h_r, fft_h_i, nodes_h = complex_matmul(transposed_r_h, transposed_i_h, h_r, h_i, "h")
        graph.nodes.extend(nodes_h)

        return fft_h_r, fft_h_i

    def _decompose_and_split(self, conv_node: gs.Node):
        input_tensor, kernel_tensor, original_output = conv_node.inputs[0], conv_node.inputs[1], conv_node.outputs[0]
        out_shape = original_output.shape
        out_h, out_w = out_shape[2], out_shape[3]
        
        kernel_data_fft = precompute_kernel_fft(kernel_tensor.values, out_shape)
        
        # --- 1. FFT Circuit (Input only) ---
        fft_graph = gs.Graph(opset=18, inputs=[input_tensor])
        
        padded_input = gs.Variable(f"{input_tensor.name}_padded", dtype=input_tensor.dtype, shape=[out_shape[0], out_shape[1], out_h, out_w])
        fft_graph.nodes.append(gs.Node(op="Pad", name="pad_input",
            inputs=[input_tensor, gs.Constant(name="pads_in", values=np.array([0,0,0,0, 0,0,out_h-input_tensor.shape[2],out_w-input_tensor.shape[3]], dtype=np.int64))],
            outputs=[padded_input]))
        
        imag_input = gs.Variable(f"{padded_input.name}_imag", dtype=padded_input.dtype, shape=padded_input.shape)
        fft_graph.nodes.append(gs.Node(op="Mul", name="create_zeros_imag", inputs=[padded_input, gs.Constant(name="const_zero", values=np.array(0, dtype=np.float32))], outputs=[imag_input]))

        w_matrix = get_dft_matrix(out_w, inverse=False)
        h_matrix = get_dft_matrix(out_h, inverse=False)
        
        fft_real_out, fft_imag_out = self._create_matmul_fft_subgraph(fft_graph, (padded_input, imag_input), w_matrix, h_matrix)

        # ** THE FIX IS HERE **: Explicitly define dtype and shape for graph outputs
        fft_real_out.dtype = original_output.dtype
        fft_imag_out.dtype = original_output.dtype
        fft_real_out.shape = out_shape
        fft_imag_out.shape = out_shape
        fft_graph.outputs = [fft_real_out, fft_imag_out]
        
        fft_path = os.path.join(self.output_dir, f"{self.base_name}_0_fft.onnx")
        onnx.save(gs.export_onnx(fft_graph), fft_path)
        self.created_files.append(fft_path)
        print(f"    ✅ Saved EZKL-compatible FFT circuit: {os.path.basename(fft_path)}")

        # --- 2. Multiply Circuit ---
        mul_in_real = gs.Variable("fft_input_real", dtype=original_output.dtype, shape=out_shape)
        mul_in_imag = gs.Variable("fft_input_imag", dtype=original_output.dtype, shape=out_shape)
        kern_fft_real = gs.Constant("kern_fft_real", np.real(kernel_data_fft).astype(np.float32))
        kern_fft_imag = gs.Constant("kern_fft_imag", np.imag(kernel_data_fft).astype(np.float32))
        mul_out_real = gs.Variable("mul_output_real", dtype=original_output.dtype, shape=out_shape)
        mul_out_imag = gs.Variable("mul_output_imag", dtype=original_output.dtype, shape=out_shape)
        
        # Create intermediate variables for the complex multiplication
        ac = gs.Variable("ac", dtype=original_output.dtype, shape=out_shape)
        bd = gs.Variable("bd", dtype=original_output.dtype, shape=out_shape)
        ad = gs.Variable("ad", dtype=original_output.dtype, shape=out_shape)
        bc = gs.Variable("bc", dtype=original_output.dtype, shape=out_shape)
        
        # Create nodes one by one to ensure proper connections
        mul_graph = gs.Graph(opset=18, inputs=[mul_in_real, mul_in_imag], outputs=[mul_out_real, mul_out_imag])
        
        # Add nodes to the graph
        mul_graph.nodes.append(gs.Node(op="Mul", inputs=[mul_in_real, kern_fft_real], outputs=[ac]))
        mul_graph.nodes.append(gs.Node(op="Mul", inputs=[mul_in_imag, kern_fft_imag], outputs=[bd]))
        mul_graph.nodes.append(gs.Node(op="Sub", inputs=[ac, bd], outputs=[mul_out_real]))
        mul_graph.nodes.append(gs.Node(op="Mul", inputs=[mul_in_real, kern_fft_imag], outputs=[ad]))
        mul_graph.nodes.append(gs.Node(op="Mul", inputs=[mul_in_imag, kern_fft_real], outputs=[bc]))
        mul_graph.nodes.append(gs.Node(op="Add", inputs=[ad, bc], outputs=[mul_out_imag]))
        
        mul_path = os.path.join(self.output_dir, f"{self.base_name}_1_mul.onnx")
        onnx.save(gs.export_onnx(mul_graph), mul_path)
        self.created_files.append(mul_path)
        print(f"    ✅ Saved EZKL-compatible Multiply circuit: {os.path.basename(mul_path)}")

        # --- 3. IFFT Circuit ---
        ifft_in_real = gs.Variable("mul_output_real", dtype=original_output.dtype, shape=out_shape)
        ifft_in_imag = gs.Variable("mul_output_imag", dtype=original_output.dtype, shape=out_shape)
        final_output = gs.Variable(original_output.name, dtype=original_output.dtype, shape=original_output.shape)
        
        ifft_graph = gs.Graph(opset=18, inputs=[ifft_in_real, ifft_in_imag], outputs=[final_output])

        w_matrix_inv = get_dft_matrix(out_w, inverse=True)
        h_matrix_inv = get_dft_matrix(out_h, inverse=True)
        
        ifft_r_out, ifft_i_out = self._create_matmul_fft_subgraph(ifft_graph, (ifft_in_real, ifft_in_imag), w_matrix_inv, h_matrix_inv)
        
        # Result is the real part of the IFFT, scaled
        ifft_graph.nodes.append(gs.Node(op="Mul", inputs=[ifft_r_out, gs.Constant("scale", np.array(1.0/(out_h*out_w), dtype=np.float32))], outputs=[final_output]))
        
        ifft_path = os.path.join(self.output_dir, f"{self.base_name}_2_ifft.onnx")
        onnx.save(gs.export_onnx(ifft_graph), ifft_path)
        self.created_files.append(ifft_path)
        print(f"    ✅ Saved EZKL-compatible IFFT circuit: {os.path.basename(ifft_path)}")


def generate_chained_metadata(all_files, output_dir):
    print("\n📋 Creating chained metadata for all generated circuits...")
    metadata = {"model_type": "EZKL_CIRCUIT_CHAIN", "description": "A model decomposed into a chain of EZKL-compatible ONNX files.", "chain_order": [os.path.basename(f) for f in all_files], "segments": []}
    for i, file_path in enumerate(all_files):
        try:
            model = onnx.load(file_path)
            segment_info = {"index": i, "name": os.path.basename(file_path), "path": file_path, "input_names": [inp.name for inp in model.graph.input], "output_names": [out.name for out in model.graph.output]}
            metadata["segments"].append(segment_info)
        except Exception as e:
            print(f"  - Warning: Could not analyze {os.path.basename(file_path)}: {e}")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to: {metadata_path}")


def main():
    base_input_dir = "./src/models/resnet/slices"
    base_output_dir = "./src/models/resnet/ezkl_circuits"

    print("🔪 ONNX Conv to EZKL-Compatible Circuit Splitter (Final Version)")
    print("=" * 70)
    
    if os.path.exists(base_output_dir): shutil.rmtree(base_output_dir)
    os.makedirs(base_output_dir)
    all_created_files = []
    
    segment_dirs = sorted([d for d in os.listdir(base_input_dir) if d.startswith("segment_")], key=lambda x: int(x.split('_')[1]))

    for segment_name in segment_dirs:
        model_path = os.path.join(base_input_dir, segment_name, f"{segment_name}.onnx")
        if not os.path.exists(model_path): continue
        
        output_dir_for_segment = os.path.join(base_output_dir, segment_name)
        try:
            splitter = ConvSplitterForEZKL(model_path, output_dir_for_segment, segment_name)
            created_files = splitter.process()
            all_created_files.extend(created_files)
        except Exception as e:
            print(f"❌ Error processing {model_path}: {e}")
            import traceback
            traceback.print_exc()

    if all_created_files: generate_chained_metadata(all_created_files, base_output_dir)
    
    print("\n🎉 Workflow complete!")
    print(f"📁 All EZKL-compatible circuits saved in: {base_output_dir}")

if __name__ == "__main__":
    main()
