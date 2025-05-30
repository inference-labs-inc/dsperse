import os.path
import json
import onnx
from src.utils.onnx_analyzer import OnnxAnalyzer
from src.utils.onnx_utils import OnnxUtils


class OnnxSlicer:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        # load onnx model
        self.onnx_model = onnx.load(onnx_path)


    #def to slice onnx model
    def slice_model(self, mode: str = None):
        if mode == "single_layer":
            # slice every layer
            self.slice_model_single_layer()
            pass
        elif mode:
            # slice 1mb chunks
            pass
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def slice_model_single_layer(self):
        """
        slice the onnx model by layer, and save the sliced model to the same directory as the original model.
        The metadata.json file will contain the input-output structure of the sliced model.
        """
        # Create output directory
        output_dir = os.path.join(os.path.dirname(self.onnx_path), "onnx_slices", "single_layers")
        os.makedirs(output_dir, exist_ok=True)

        # Store node metadata temporarily
        node_metadata = {}
        graph = self.onnx_model.graph
        print(f"Number of layers in the model: {len(graph.node)}")

        # Create maps for initializers and value info
        initializer_map = {init.name: init for init in graph.initializer}

        # Build a comprehensive value_info map from the original full model
        full_model_value_info_map = {vi.name: vi for vi in graph.value_info}
        full_model_value_info_map.update({vi.name: vi for vi in graph.input})
        full_model_value_info_map.update({vi.name: vi for vi in graph.output})

        # Create an analyzer for the model
        analyzer = OnnxAnalyzer(onnx_model=self.onnx_model, onnx_path=self.onnx_path)

        # Process each node
        for i, node in enumerate(graph.node):
            # Find which inputs are not produced by other nodes (= need as graph inputs)
            all_outputs = {o for n in graph.node for o in n.output}
            external_inputs = [inp for inp in node.input if inp not in all_outputs]

            # Construct input tensors (graph.input) for the single-node graph
            actual_inputs = []
            for inp in node.input:
                if inp in full_model_value_info_map:
                    actual_inputs.append(full_model_value_info_map[inp])
                else:
                    t = onnx.helper.make_tensor_value_info(inp, onnx.TensorProto.FLOAT, [None])
                    actual_inputs.append(t)

            # Initializers needed by this node
            node_initializers = [initializer_map[inp] for inp in node.input if inp in initializer_map]

            # Output value_info:
            actual_outputs = []
            for out in node.output:
                if out in full_model_value_info_map:
                    actual_outputs.append(full_model_value_info_map[out])
                else:
                    t = onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, [None])
                    actual_outputs.append(t)

            # Create and save the single-node model
            model = OnnxUtils.create_node_model(node, actual_inputs, actual_outputs, node_initializers)
            save_path = os.path.join(output_dir, f"segment_{i}.onnx")
            OnnxUtils.save_model(model, save_path)

            # Analyze the node and store metadata
            node_info = analyzer.analyze_node(node, i, initializer_map, full_model_value_info_map)
            node_info["path"] = save_path  # Update the path in the metadata
            node_metadata[node.name] = node_info

        # Create segments from node metadata
        segments, total_parameters = analyzer.create_segments_from_metadata(node_metadata)

        # Generate metadata without shape_info
        metadata = analyzer.generate_metadata(segments, output_dir)
        if metadata["original_model"] is None:
            metadata["original_model"] = self.onnx_path

        # Save metadata
        OnnxUtils.save_metadata(metadata, output_dir)

if __name__ == "__main__":

    model_choice = 1  # Change this to test different models

    base_paths = {
        1: "models/doom",
        2: "models/net",
        3: "models/resnet",
        4: "models/yolov3"
    }

    model_dir = os.path.join(base_paths[model_choice], "model.onnx")
    onnx_slicer = OnnxSlicer(model_dir)

    onnx_slicer.slice_model(mode="single_layer")
