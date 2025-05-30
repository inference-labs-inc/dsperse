import os
import json
import onnx
from onnx import shape_inference

class OnnxUtils:
    """
    Utility functions for working with ONNX models.
    """
    
    @staticmethod
    def save_metadata(metadata, output_dir):
        """
        Save metadata to a JSON file.
        
        Args:
            metadata: Dictionary containing metadata
            output_dir: Directory where the metadata will be saved
        """
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    @staticmethod
    def create_node_model(node, inputs, outputs, initializers):
        """
        Create a minimal ONNX model containing a single node.
        
        Args:
            node: ONNX node to include in the model
            inputs: List of input tensor value infos
            outputs: List of output tensor value infos
            initializers: List of initializers needed by the node
            
        Returns:
            onnx.ModelProto: ONNX model containing the node
        """
        # Create a graph with the node
        graph = onnx.helper.make_graph(
            [node],  # single node
            f"{node.name}_graph",
            inputs,
            outputs,
            initializers
        )
        
        # Create a model from the graph
        model = onnx.helper.make_model(graph)
        
        return model
    
    @staticmethod
    def infer_shapes(model):
        """
        Infer shapes for an ONNX model.
        
        Args:
            model: ONNX model
            
        Returns:
            onnx.ModelProto: ONNX model with inferred shapes
        """
        try:
            inferred_model = shape_inference.infer_shapes(model)
            return inferred_model
        except Exception as e:
            print(f"Warning: Shape inference failed: {e}")
            return model
    
    @staticmethod
    def save_model(model, path):
        """
        Save an ONNX model to a file.
        
        Args:
            model: ONNX model
            path: Path where the model will be saved
        """
        onnx.save(model, path)
    
    @staticmethod
    def get_tensor_value_info(name, dtype, shape):
        """
        Create a tensor value info.
        
        Args:
            name: Name of the tensor
            dtype: Data type of the tensor (e.g., onnx.TensorProto.FLOAT)
            shape: Shape of the tensor
            
        Returns:
            onnx.ValueInfoProto: Tensor value info
        """
        return onnx.helper.make_tensor_value_info(name, dtype, shape)
    
    @staticmethod
    def extract_shape_from_value_info(value_info):
        """
        Extract shape from a value info.
        
        Args:
            value_info: ONNX value info
            
        Returns:
            list: Shape of the tensor
        """
        shape = []
        if value_info.type.tensor_type.shape.dim:
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value if dim.dim_value != 0 else None)
        return shape
    
    @staticmethod
    def get_output_dir_for_slices(onnx_path):
        """
        Get the output directory for sliced models.
        
        Args:
            onnx_path: Path to the original ONNX model
            
        Returns:
            str: Path to the output directory
        """
        return os.path.join(os.path.dirname(onnx_path), "onnx_slices", "single_layers")