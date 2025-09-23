import torch
import torch.nn as nn
import ast
import importlib.util
import sys
import os


class ModelAnalyzer:
    def __init__(self, model=None):
        self.model = model
        self.model_type = None
        self.analysis_results = {}

    def get_activation_functions(self, model_path=None, model_file=None, class_name=None):
        """
        Extract all activation functions used in a PyTorch model by analyzing both
        the loaded model and the source code.

        Args:
            model_path: Path to the .pth file containing the model weights
            model_file: Path to the Python file containing the model definition
            class_name: Name of the model class to analyze

        Returns:
            A dictionary mapping layer names to their activation functions
        """

        activations = {}

        # PART 1: Extract explicit activation modules from the loaded model if model_path is provided
        if model_path:
            try:
                # We need to load the model class first from the Python file
                if model_file and class_name:
                    # Get the directory of the model file
                    model_dir = os.path.dirname(os.path.abspath(model_file))

                    # Add the directory to sys.path if it's not already there
                    if model_dir not in sys.path:
                        sys.path.insert(0, model_dir)

                    # Get the module name without .py extension
                    module_name = os.path.basename(model_file).replace('.py', '')

                    # Load the module from file
                    spec = importlib.util.spec_from_file_location(module_name, model_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Get the model class
                    model_class = getattr(module, class_name)

                    # Load the model weights
                    model = model_class()
                    model.load_state_dict(torch.load(model_path))
                    model.eval()  # Set model to evaluation mode

                    # Define activation classes to look for
                    activation_classes = (
                        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ELU, nn.SELU, nn.CELU, nn.GELU, nn.Sigmoid,
                        nn.Tanh, nn.SiLU, nn.Mish, nn.Hardswish, nn.Hardtanh, nn.Hardsigmoid, nn.Softmax,
                        nn.LogSoftmax, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanhshrink
                    )

                    # Track which modules we've seen to handle potential recursion
                    visited = set()

                    def _extract_explicit_activations(module, path=''):
                        """Recursively extract activations from module and its children."""
                        # Skip if we've already processed this module
                        if id(module) in visited:
                            return
                        visited.add(id(module))

                        # Check if the current module is an activation function
                        if isinstance(module, activation_classes):
                            act_type = module.__class__.__name__
                            params = {}

                            # Extract parameters that aren't the default values
                            for name, param in module.__dict__.items():
                                if not name.startswith('_'):  # Skip private attributes
                                    params[name] = param

                            activations[path] = {
                                'type': 'explicit',
                                'class': act_type,
                                'params': params
                            }

                        # Recursively process child modules
                        for name, child in module.named_children():
                            child_path = f"{path}.{name}" if path else name
                            _extract_explicit_activations(child, child_path)

                    # Start recursive extraction for explicit activations
                    _extract_explicit_activations(model)
                else:
                    print("Need both model_file and class_name to load the PyTorch model")

            except Exception as e:
                print(f"Error loading or analyzing PyTorch model: {e}")

        # PART 2: Extract functional activations from source code
        if model_file and class_name:
            try:
                with open(model_file, 'r') as f:
                    source = f.read()
                tree = ast.parse(source)

                # Create containers for extract_forward_flow results
                layer_order = []
                functional_activations = {}
                reshape_ops = {}

                # Call extract_forward_flow with correct self reference
                self._extract_forward_flow_inner(tree, class_name, layer_order, functional_activations, reshape_ops)

                # Add functional activations to the results
                for layer, info in functional_activations.items():
                    if layer in activations:
                        # If we already found this layer from the explicit model,
                        # add the functional info as additional info
                        activations[layer]['functional_info'] = info
                    else:
                        # Otherwise add as a new entry
                        activations[layer] = {
                            'type': 'functional',
                            'info': info
                        }

            except Exception as e:
                print(f"Error analyzing model source code: {e}")

        return activations

    def break_model_into_segments(self, model_file: str, class_name: str):
        """Break a model into separate layer segment modules."""
        # Parse the source code
        with open(model_file, 'r') as f:
            source = f.read()
        tree = ast.parse(source)

        # Extract information from the model
        layers_info = ModelAnalyzer.extract_layers_from_init(tree, class_name)
        layer_order, activations, reshape_ops = self.extract_forward_flow(tree, class_name)

        # Generate segment classes
        segment_classes = {}
        for layer_name in layer_order:
            if layer_name.startswith('conv'):
                segment_classes[f"{layer_name.capitalize()}Segment"] = ModelAnalyzer.create_conv_segment(
                    layer_name, layers_info[layer_name], activations.get(layer_name))
            elif layer_name.startswith('fc'):
                reshape_code = ModelAnalyzer.extract_reshape_code(tree, class_name, layer_name)
                segment_classes[f"{layer_name.capitalize()}Segment"] = ModelAnalyzer.create_linear_segment(
                    layer_name, layers_info[layer_name], activations.get(layer_name),
                    reshape_code)
            elif layer_name == 'pool':
                segment_classes[f"{layer_name.capitalize()}Segment"] = ModelAnalyzer.create_pool_segment(
                    layer_name, layers_info[layer_name], activations.get(layer_name))

        return segment_classes

    @staticmethod
    def extract_layers_from_init(tree, class_name):
        """Extract layer definitions from __init__ method."""
        layers_info = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        for stmt in method.body:
                            if (isinstance(stmt, ast.Assign) and
                                    isinstance(stmt.targets[0], ast.Attribute) and
                                    isinstance(stmt.targets[0].value, ast.Name) and
                                    stmt.targets[0].value.id == 'self'):

                                attr_name = stmt.targets[0].attr
                                if isinstance(stmt.value, ast.Call):
                                    # Extract layer definitions
                                    if attr_name in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2', 'fc3', 'pool']:
                                        layers_info[attr_name] = ast.unparse(stmt.value)
                                # Store other attributes that might be needed for reshape operations
                                else:
                                    layers_info[attr_name] = ast.unparse(stmt.value)

        return layers_info

    def _extract_forward_flow_inner(self, tree, class_name, layer_order, activations, reshape_ops):
        """Helper method that implements extract_forward_flow with explicit arguments."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "forward":
                        # Analyze forward method body
                        for i, stmt in enumerate(method.body):
                            # Check for activations with layers
                            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                                self.analyze_assignment(stmt, layer_order, activations, reshape_ops)

                            # Check for nested calls like pool(relu(conv))
                            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                                self.analyze_return_statement(stmt, layer_order, activations)


    def extract_forward_flow(self, tree, class_name):
        """Extract layer execution order, activations, and reshape operations from forward method."""
        layer_order = []
        activations = {}
        reshape_ops = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "forward":
                        # Analyze forward method body
                        for i, stmt in enumerate(method.body):
                            # Check for activations with layers
                            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                                self.analyze_assignment(stmt, layer_order, activations, reshape_ops)

                            # Check for nested calls like pool(relu(conv))
                            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
                                self.analyze_return_statement(stmt, layer_order, activations)

        return layer_order, activations, reshape_ops

    @staticmethod
    def analyze_assignment(stmt, layer_order, activations, reshape_ops):
        """Analyze an assignment statement in the forward method."""
        def extract_layer_from_call(call_node):
            """Extract layer name from a call node."""
            if (isinstance(call_node, ast.Call) and
                    isinstance(call_node.func, ast.Attribute) and
                    isinstance(call_node.func.value, ast.Name) and
                    call_node.func.value.id == 'self'):
                return call_node.func.attr
            return None

        def extract_activation_from_call(call_node):
            """Extract activation function name from a call node."""
            if (isinstance(call_node, ast.Call) and
                    isinstance(call_node.func, ast.Attribute) and
                    isinstance(call_node.func.value, ast.Name) and
                    call_node.func.value.id == 'F'):
                return call_node.func.attr
            return None

        # Check for F.activation(self.layer(x)) pattern
        if isinstance(stmt.value.func, ast.Attribute):
            activation = extract_activation_from_call(stmt.value)
            if activation:
                if stmt.value.args and isinstance(stmt.value.args[0], ast.Call):
                    layer = extract_layer_from_call(stmt.value.args[0])
                    if layer and layer not in layer_order:
                        layer_order.append(layer)
                        activations[layer] = activation

        # Check for reshape operations - look for torch.flatten or x.reshape
        elif isinstance(stmt.value, ast.Call):
            if ((isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr in ['reshape', 'view']) or
                    (isinstance(stmt.value.func, ast.Attribute) and
                     isinstance(stmt.value.func.value, ast.Name) and
                     stmt.value.func.value.id == 'torch' and
                     stmt.value.func.attr == 'flatten')):
                reshape_ops['reshape_before_next'] = True

        # Check for self.layer(x) pattern
        else:
            layer = extract_layer_from_call(stmt.value)
            if layer and layer not in layer_order:
                layer_order.append(layer)

    @staticmethod
    def analyze_return_statement(stmt, layer_order, activations):
        """Analyze a return statement for layer order and activations."""
        def process_call(call_node):
            """Process a call node to extract layers and activations."""
            if isinstance(call_node, ast.Call):
                if (isinstance(call_node.func, ast.Attribute) and
                        isinstance(call_node.func.value, ast.Name)):

                    if call_node.func.value.id == 'F' and call_node.args:
                        # F.activation(arg)
                        activation = call_node.func.attr
                        if isinstance(call_node.args[0], ast.Call):
                            process_call(call_node.args[0], activation)

                    elif call_node.func.value.id == 'self':
                        # self.layer(arg)
                        layer = call_node.func.attr
                        if layer not in layer_order:
                            layer_order.append(layer)

                        # If we know the activation for this layer
                        if 'current_activation' in locals():
                            activations[layer] = locals()['current_activation']

        process_call(stmt.value)

    @staticmethod
    def extract_reshape_code(tree, class_name, layer_name):
        """Extract reshape/flatten code used before a specific layer."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "forward":
                        for i, stmt in enumerate(method.body):
                            # Look for reshape/flatten operations before the layer
                            if i + 1 < len(method.body):
                                current_stmt = stmt
                                next_stmt = method.body[i + 1]

                                # Check if current is reshape and next uses our layer
                                if (isinstance(current_stmt, ast.Assign) and
                                        isinstance(current_stmt.value, ast.Call) and
                                        ((isinstance(current_stmt.value.func, ast.Attribute) and
                                          current_stmt.value.func.attr in ['reshape', 'view']) or
                                         (isinstance(current_stmt.value.func, ast.Attribute) and
                                          isinstance(current_stmt.value.func.value, ast.Name) and
                                          current_stmt.value.func.value.id == 'torch' and
                                          current_stmt.value.func.attr == 'flatten'))):

                                    # Check if next statement uses our layer
                                    if ((isinstance(next_stmt, ast.Assign) and
                                         isinstance(next_stmt.value, ast.Call) and
                                         isinstance(next_stmt.value.func, ast.Attribute) and
                                         'F.relu' in ast.unparse(next_stmt.value.func) and
                                         layer_name in ast.unparse(next_stmt.value))):
                                        reshape_code = ast.unparse(current_stmt.value)
                                        return reshape_code

        return None

    @staticmethod
    def create_conv_segment(layer_name, layer_def, activation=None):
        """Create a segment class for a convolutional layer."""
        class_name = f"{layer_name.capitalize()}Segment"

        class_def = f"class {class_name}(nn.Module):\n"
        class_def += "    def __init__(self):\n"
        class_def += f"        super({class_name}, self).__init__()\n"
        class_def += f"        self.{layer_name} = {layer_def}\n\n"

        class_def += "    def forward(self, x):\n"
        if activation:
            class_def += f"        return F.{activation}(self.{layer_name}(x))\n"
        else:
            class_def += f"        return self.{layer_name}(x)\n"

        return class_def

    @staticmethod
    def create_linear_segment(layer_name, layer_def, activation=None, reshape_code=None):
        """Create a segment class for a linear layer."""
        class_name = f"{layer_name.capitalize()}Segment"

        class_def = f"class {class_name}(nn.Module):\n"
        class_def += "    def __init__(self):\n"
        class_def += f"        super({class_name}, self).__init__()\n"
        class_def += f"        self.{layer_name} = {layer_def}\n\n"

        class_def += "    def forward(self, x):\n"

        # Add reshape if needed
        if reshape_code:
            if 'flatten' in reshape_code:
                class_def += "        x = torch.flatten(x, 1)\n"
            else:
                # Extract just the reshape dimensions, not the variable name
                reshape_parts = reshape_code.split('reshape(')[1].split(')')[0]
                class_def += f"        x = x.reshape({reshape_parts})\n"

        if activation:
            class_def += f"        return F.{activation}(self.{layer_name}(x))\n"
        else:
            class_def += f"        return self.{layer_name}(x)\n"

        return class_def

    @staticmethod
    def create_pool_segment(layer_name, layer_def, activation=None):
        """Create a segment class for a pooling layer."""
        class_name = f"{layer_name.capitalize()}Segment"

        class_def = f"class {class_name}(nn.Module):\n"
        class_def += "    def __init__(self):\n"
        class_def += f"        super({class_name}, self).__init__()\n"
        class_def += f"        self.{layer_name} = {layer_def}\n\n"

        class_def += "    def forward(self, x):\n"
        if activation:
            class_def += f"        return F.{activation}(self.{layer_name}(x))\n"
        else:
            class_def += f"        return self.{layer_name}(x)\n"

        return class_def


# Example usage:
if __name__ == "__main__":
    model_path = "../models/net/model.py"
    torch_path = "../models/net/model.pth"
    analyzer = ModelAnalyzer(model_path)

    segment_classes = analyzer.break_model_into_segments(model_path, "Net")
    print(segment_classes)
    # activation_functions = analyzer.get_activation_functions(torch_path, model_path, "Net")
    # print(activation_functions)

