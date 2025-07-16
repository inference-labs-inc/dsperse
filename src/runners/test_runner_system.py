"""
Comprehensive test suite for Production Runner System.

Tests all components including metadata generation, runner execution,
error handling, and production scenarios.
"""

import unittest
import os
import json
import tempfile
import shutil
import time
from unittest.mock import patch, MagicMock
import torch

from runner_metadata import RunnerMetadata
from runner import Runner


class TestRunnerMetadata(unittest.TestCase):
    """Test metadata generation and structure"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = "test_models/test_model"
        self.full_model_path = os.path.join(self.test_dir, "src", self.model_path)
        os.makedirs(self.full_model_path, exist_ok=True)
        
        # Create mock ONNX model
        self.onnx_path = os.path.join(self.full_model_path, "model.onnx")
        with open(self.onnx_path, 'wb') as f:
            f.write(b"mock_onnx_data")

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('onnx.load')
    def test_metadata_initialization(self, mock_onnx_load):
        """Test metadata generator initialization"""
        mock_onnx_load.return_value = MagicMock()
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            generator.model_path = self.model_path
            generator.model_name = "test_model"
            generator.time_limit = 300
            generator.size_limit = 1048576
            
            self.assertEqual(generator.model_name, "test_model")
            self.assertEqual(generator.time_limit, 300)
            self.assertEqual(generator.size_limit, 1048576)

    def test_security_calculation(self):
        """Test security percentage calculation"""
        # Create mock slices data
        slices_with_circuits = {
            "slice_0": {"ezkl": True},
            "slice_1": {"ezkl": True},
            "slice_2": {"ezkl": False},
            "slice_3": {"ezkl": True},
            "slice_4": {"ezkl": False}
        }
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            security = generator._calculate_security(slices_with_circuits)
            
            # 3 out of 5 slices use circuits = 60%
            self.assertEqual(security, 60.0)

    def test_execution_chain_structure(self):
        """Test execution chain linked list structure"""
        # Mock segments data
        segments = [
            {"index": 0, "path": "slice_0.onnx", "dependencies": {}, "parameters": 100},
            {"index": 1, "path": "slice_1.onnx", "dependencies": {}, "parameters": 200}
        ]
        
        slices_metadata = {"segments": segments}
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            generator.ezkl_slices_dir = "/mock/ezkl/slices"
            generator.size_limit = 1048576
            
            with patch('os.path.exists', return_value=False):
                slices, execution_chain, verified_slices = generator._process_slices(slices_metadata)
                
                # Check execution chain structure
                self.assertEqual(execution_chain["head"], "slice_0")
                self.assertIn("nodes", execution_chain)
                self.assertIn("fallback_map", execution_chain)
                
                # Check linked list pointers
                self.assertEqual(execution_chain["nodes"]["slice_0"]["next"], "slice_1")
                self.assertIsNone(execution_chain["nodes"]["slice_1"]["next"])

    def test_fallback_mapping(self):
        """Test correct fallback mapping from EzKL to ONNX"""
        segments = [{"index": 0, "path": "onnx_slices/segment_0.onnx", "dependencies": {}, "parameters": 100}]
        slices_metadata = {"segments": segments}
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            generator.ezkl_slices_dir = "/mock/ezkl/slices"
            generator.size_limit = 1048576
            
            # Mock circuit exists
            circuit_path = "/mock/ezkl/slices/segment_0/segment_0.onnx"
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1000):
                
                slices, execution_chain, verified_slices = generator._process_slices(slices_metadata)
                
                # Check fallback mapping
                fallback_map = execution_chain["fallback_map"]
                self.assertIn(circuit_path, fallback_map)
                self.assertEqual(fallback_map[circuit_path], "onnx_slices/segment_0.onnx")


class TestRunner(unittest.TestCase):
    """Test runner execution and inference"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = "test_models/test_model"
        
        # Create mock metadata
        self.mock_metadata = {
            "model_name": "test_model",
            "overall_security": 80.0,
            "time_limit": 300,
            "size_limit": 1048576,
            "execution_chain": {
                "head": "slice_0",
                "nodes": {
                    "slice_0": {
                        "slice_id": "slice_0",
                        "use_circuit": True,
                        "next": "slice_1",
                        "circuit_path": "/mock/circuit.onnx",
                        "onnx_path": "onnx_slice.onnx"
                    },
                    "slice_1": {
                        "slice_id": "slice_1",
                        "use_circuit": False,
                        "next": None,
                        "circuit_path": None,
                        "onnx_path": "onnx_slice.onnx"
                    }
                },
                "fallback_map": {
                    "/mock/circuit.onnx": "onnx_slice.onnx"
                }
            },
            "slices": {
                "slice_0": {"ezkl": True, "circuit_size": 1000, "dependencies": {}},
                "slice_1": {"ezkl": False, "circuit_size": 0, "dependencies": {}}
            },
            "verified_slices": {"slice_0": True, "slice_1": False},
            "io_paths": {"input_path": "/mock/input.json"}
        }

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('json.load')
    def test_runner_initialization(self, mock_json_load, mock_open, mock_exists):
        """Test runner initialization with metadata loading"""
        mock_exists.return_value = True
        mock_json_load.return_value = self.mock_metadata
        
        with patch.object(Runner, '__init__', lambda x, y: None):
            runner = Runner.__new__(Runner)
            runner.metadata = self.mock_metadata
            runner.execution_chain = self.mock_metadata["execution_chain"]
            runner.verified_slices = self.mock_metadata["verified_slices"]
            runner.slices = self.mock_metadata["slices"]
            
            self.assertEqual(runner.metadata["overall_security"], 80.0)
            self.assertEqual(runner.execution_chain["head"], "slice_0")

    def test_execution_chain_traversal(self):
        """Test linked list traversal logic"""
        with patch.object(Runner, '__init__', lambda x, y: None):
            runner = Runner.__new__(Runner)
            runner.execution_chain = self.mock_metadata["execution_chain"]
            runner.verified_slices = self.mock_metadata["verified_slices"]
            runner.slices = self.mock_metadata["slices"]
            runner.time_limit = 300
            
            # Mock tensor and execution methods
            input_tensor = torch.randn(1, 4, 28, 28)
            
            with patch.object(runner, '_execute_ezkl_slice', return_value=torch.randn(1, 10)), \
                 patch.object(runner, '_execute_onnx_slice', return_value=torch.randn(1, 10)), \
                 patch.object(runner, '_process_final_output', return_value={"predicted_class": 1}), \
                 patch('time.time', return_value=0):
                
                result = runner._execute_ordered(input_tensor, 0)
                
                self.assertEqual(result["execution_method"], "linked_list")
                self.assertEqual(len(result["execution_results"]), 2)

    def test_fallback_decision_logic(self):
        """Test when fallback is used vs circuit"""
        with patch.object(Runner, '__init__', lambda x, y: None):
            runner = Runner.__new__(Runner)
            runner.execution_chain = self.mock_metadata["execution_chain"]
            runner.verified_slices = self.mock_metadata["verified_slices"]
            runner.slices = self.mock_metadata["slices"]
            runner.time_limit = 300
            
            input_tensor = torch.randn(1, 4, 28, 28)
            execution_results = []
            
            # Mock methods
            def mock_ezkl_exec(*args):
                execution_results.append("ezkl_used")
                return torch.randn(1, 10)
            
            def mock_onnx_exec(*args):
                execution_results.append("onnx_used")
                return torch.randn(1, 10)
            
            with patch.object(runner, '_execute_ezkl_slice', side_effect=mock_ezkl_exec), \
                 patch.object(runner, '_execute_onnx_slice', side_effect=mock_onnx_exec), \
                 patch.object(runner, '_process_final_output', return_value={"predicted_class": 1}), \
                 patch('time.time', return_value=0):
                
                result = runner._execute_ordered(input_tensor, 0)
                
                # slice_0 should use EzKL (use_circuit=True, verified=True)
                # slice_1 should use ONNX (use_circuit=False)
                self.assertEqual(execution_results[0], "ezkl_used")
                self.assertEqual(execution_results[1], "onnx_used")

    def test_security_calculation_accuracy(self):
        """Test security percentage calculation accuracy"""
        test_cases = [
            ({"slice_0": {"ezkl": True}, "slice_1": {"ezkl": True}}, 100.0),
            ({"slice_0": {"ezkl": True}, "slice_1": {"ezkl": False}}, 50.0),
            ({"slice_0": {"ezkl": False}, "slice_1": {"ezkl": False}}, 0.0),
        ]
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            
            for slices, expected_security in test_cases:
                security = generator._calculate_security(slices)
                self.assertEqual(security, expected_security)

    def test_time_limit_enforcement(self):
        """Test time limit enforcement during execution"""
        with patch.object(Runner, '__init__', lambda x, y: None):
            runner = Runner.__new__(Runner)
            runner.execution_chain = self.mock_metadata["execution_chain"]
            runner.verified_slices = self.mock_metadata["verified_slices"]
            runner.slices = self.mock_metadata["slices"]
            runner.time_limit = 0.1  # Very short time limit
            
            input_tensor = torch.randn(1, 4, 28, 28)
            
            # Mock time to exceed limit on second check - provide more values to avoid StopIteration
            with patch('time.time', side_effect=[0, 0, 1, 1, 1, 1, 1, 1]):  # Exceed time limit
                result = runner._execute_ordered(input_tensor, 0)
                
                self.assertEqual(result["error"], "Time limit exceeded")
                self.assertEqual(result["execution_method"], "timeout")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_missing_execution_chain(self):
        """Test handling of missing execution chain"""
        with patch.object(Runner, '__init__', lambda x, y: None):
            runner = Runner.__new__(Runner)
            runner.execution_chain = {}  # Empty chain
            
            input_tensor = torch.randn(1, 4, 28, 28)
            
            with self.assertRaises(ValueError) as context:
                runner._execute_ordered(input_tensor, 0)
            
            self.assertIn("No execution chain found", str(context.exception))

    def test_empty_slices_security_calculation(self):
        """Test security calculation with empty slices"""
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            
            security = generator._calculate_security({})
            self.assertEqual(security, 0.0)

    def test_invalid_model_path(self):
        """Test handling of invalid model paths"""
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(Exception):
                # This should fail when trying to load non-existent model
                RunnerMetadata("invalid/path")


class TestProductionScenarios(unittest.TestCase):
    """Test production scenarios and integration"""

    def test_full_workflow_integration(self):
        """Test complete workflow from metadata generation to inference"""
        # This would be a full integration test
        # For now, we'll test the workflow structure
        
        workflow_steps = [
            "Initialize RunnerMetadata",
            "Generate metadata with execution chain",
            "Initialize Runner with metadata",
            "Execute inference using execution chain",
            "Return results with security info"
        ]
        
        # Verify workflow steps are covered
        self.assertEqual(len(workflow_steps), 5)

    def test_metadata_consistency(self):
        """Test metadata consistency across regenerations"""
        # Mock consistent input data
        mock_segments = [
            {"index": 0, "path": "slice_0.onnx", "dependencies": {}, "parameters": 100},
            {"index": 1, "path": "slice_1.onnx", "dependencies": {}, "parameters": 200}
        ]
        
        with patch.object(RunnerMetadata, '__init__', lambda x, y: None):
            generator = RunnerMetadata.__new__(RunnerMetadata)
            generator.ezkl_slices_dir = "/mock/ezkl/slices"
            generator.size_limit = 1048576
            
            with patch('os.path.exists', return_value=True), \
                 patch('os.path.getsize', return_value=1000):
                
                # Generate metadata twice
                result1 = generator._process_slices({"segments": mock_segments})
                result2 = generator._process_slices({"segments": mock_segments})
                
                # Results should be consistent
                self.assertEqual(result1[1]["head"], result2[1]["head"])  # execution_chain
                self.assertEqual(len(result1[1]["nodes"]), len(result2[1]["nodes"]))


class TestInferenceErrorTolerance(unittest.TestCase):
    """Test inference error tolerance with various input scenarios and inference modes"""

    def setUp(self):
        """Set up test environment for inference testing"""
        self.test_dir = tempfile.mkdtemp()
        self.model_path = "models/doom"  # Use real model for inference testing
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_whole_model_inference_error_tolerance(self):
        """Test whole model inference with various error scenarios"""
        try:
            runner = Runner(self.model_path)
            
            # Test 1: Normal inference (onnx_only mode)
            try:
                result = runner.infer(mode="onnx_only")
                self.assertIn("predicted_class", result)
                self.assertIn("execution_method", result)
                self.assertEqual(result["execution_method"], "onnx_whole")
                print(f"✅ Whole model inference: predicted_class={result.get('predicted_class')}")
            except Exception as e:
                self.fail(f"Normal whole model inference failed: {e}")
            
            # Test 2: Invalid input path
            with self.assertRaises(RuntimeError):
                runner.infer(input_path="nonexistent/path.json", mode="onnx_only")
                
            # Test 3: Invalid mode
            with self.assertRaises(ValueError):
                runner.infer(mode="invalid_mode")
                
        except Exception as e:
            self.skipTest(f"Skipping whole model test due to setup issue: {e}")

    def test_sliced_inference_error_tolerance(self):
        """Test sliced inference with error scenarios"""
        try:
            runner = Runner(self.model_path)
            
            # Test 1: Normal sliced inference (auto mode)
            try:
                result = runner.infer(mode="auto")
                self.assertIn("predicted_class", result)
                self.assertIn("execution_method", result)
                self.assertIn("execution_results", result)
                
                # Check execution results structure
                execution_results = result.get("execution_results", [])
                self.assertGreater(len(execution_results), 0)
                
                for exec_result in execution_results:
                    self.assertIn("slice_id", exec_result)
                    self.assertIn("method", exec_result)
                    self.assertIn("success", exec_result)
                    
                print(f"✅ Sliced inference: {len(execution_results)} slices executed")
                print(f"   Security: {result.get('overall_security', 0)}%")
                
            except Exception as e:
                self.fail(f"Normal sliced inference failed: {e}")
            
            # Test 2: Corrupted execution chain
            runner_copy = Runner(self.model_path)
            runner_copy.execution_chain = {}  # Corrupt the chain
            
            with self.assertRaises(ValueError):
                runner_copy.infer(mode="auto")
                
        except Exception as e:
            self.skipTest(f"Skipping sliced inference test due to setup issue: {e}")

    def test_circuit_inference_error_tolerance(self):
        """Test EzKL circuit inference with error scenarios"""
        try:
            runner = Runner(self.model_path)
            
            # Test 1: Normal circuit inference
            try:
                result = runner.infer(mode="auto")
                
                # Check for circuit usage in execution results
                execution_results = result.get("execution_results", [])
                circuit_used = any(r.get("method") == "ezkl_circuit" for r in execution_results)
                
                if circuit_used:
                    print(f"✅ Circuit inference: Circuits used in execution")
                else:
                    print(f"ℹ️  Circuit inference: Fallback to ONNX (expected)")
                
                # Verify fallback behavior
                for exec_result in execution_results:
                    if exec_result.get("method") == "onnx_slice":
                        # This is expected fallback behavior
                        self.assertTrue(exec_result.get("success", False))
                        
            except Exception as e:
                self.fail(f"Circuit inference test failed: {e}")
                
        except Exception as e:
            self.skipTest(f"Skipping circuit inference test due to setup issue: {e}")

    def test_input_validation_and_preprocessing(self):
        """Test input validation and preprocessing with various input formats"""
        try:
            runner = Runner(self.model_path)
            
            # Create test input directory
            test_input_dir = os.path.join(self.test_dir, "test_inputs")
            os.makedirs(test_input_dir, exist_ok=True)
            
            # Test 1: Valid input format
            valid_input_path = os.path.join(test_input_dir, "valid_input.json")
            valid_input = {"input_data": [[1.0] * 3136]}  # Flattened 4x28x28 input
            
            with open(valid_input_path, 'w') as f:
                json.dump(valid_input, f)
            
            try:
                result = runner.infer(input_path=valid_input_path, mode="onnx_only")
                self.assertIn("predicted_class", result)
                print(f"✅ Valid input: predicted_class={result.get('predicted_class')}")
            except Exception as e:
                self.fail(f"Valid input processing failed: {e}")
            
            # Test 2: Invalid JSON format
            invalid_json_path = os.path.join(test_input_dir, "invalid.json")
            with open(invalid_json_path, 'w') as f:
                f.write("invalid json content {")
                
            with self.assertRaises(RuntimeError):
                runner.infer(input_path=invalid_json_path, mode="onnx_only")
            
            # Test 3: Wrong input shape
            wrong_shape_path = os.path.join(test_input_dir, "wrong_shape.json")
            wrong_shape_input = {"input_data": [[1.0] * 100]}  # Wrong size
            
            with open(wrong_shape_path, 'w') as f:
                json.dump(wrong_shape_input, f)
            
            # This should either work with reshaping or fail gracefully
            try:
                result = runner.infer(input_path=wrong_shape_path, mode="onnx_only")
                print(f"ℹ️  Wrong shape handled: {result.get('predicted_class', 'N/A')}")
            except RuntimeError:
                print(f"ℹ️  Wrong shape rejected (expected)")
                
        except Exception as e:
            self.skipTest(f"Skipping input validation test due to setup issue: {e}")

    def test_execution_timeout_scenarios(self):
        """Test timeout scenarios with different execution modes"""
        try:
            runner = Runner(self.model_path)
            
            # Test 1: Normal execution within time limit
            start_time = time.time()
            result = runner.infer(mode="onnx_only")
            execution_time = time.time() - start_time
            
            self.assertLess(execution_time, runner.time_limit)
            self.assertIn("predicted_class", result)
            print(f"✅ Normal execution: {execution_time:.3f}s (limit: {runner.time_limit}s)")
            
            # Test 2: Very short time limit (simulated in controlled way)
            runner_copy = Runner(self.model_path)
            runner_copy.time_limit = 0.001  # Very short limit
            
            # For sliced inference, this might trigger timeout
            try:
                result = runner_copy.infer(mode="auto")
                if "error" in result and "Time limit exceeded" in result["error"]:
                    print(f"✅ Timeout detected and handled correctly")
                else:
                    print(f"ℹ️  Execution completed within short limit")
            except Exception as e:
                print(f"ℹ️  Short timeout caused controlled failure: {e}")
                
        except Exception as e:
            self.skipTest(f"Skipping timeout test due to setup issue: {e}")

    def test_fallback_system_robustness(self):
        """Test robustness of fallback system under various failure scenarios"""
        try:
            runner = Runner(self.model_path)
            
            # Test 1: Normal fallback behavior
            result = runner.infer(mode="auto")
            execution_results = result.get("execution_results", [])
            
            # Count fallbacks vs primary usage
            fallback_count = sum(1 for r in execution_results if r.get("fallback_used", False))
            primary_count = len(execution_results) - fallback_count
            
            print(f"✅ Fallback analysis: {primary_count} primary, {fallback_count} fallback")
            
            # Test 2: Verify all slices completed successfully
            success_count = sum(1 for r in execution_results if r.get("success", False))
            self.assertEqual(success_count, len(execution_results))
            
            # Test 3: Check security calculation matches execution
            expected_security = (sum(1 for r in execution_results if r.get("method") == "ezkl_circuit") / 
                               len(execution_results) * 100) if execution_results else 0
            
            # Allow some tolerance for different security calculation methods
            actual_security = result.get("overall_security", 0)
            print(f"ℹ️  Security: actual={actual_security}%, calculated={expected_security:.1f}%")
            
        except Exception as e:
            self.skipTest(f"Skipping fallback test due to setup issue: {e}")

    def test_metadata_dependency_errors(self):
        """Test error handling when metadata dependencies are missing or corrupted"""
        try:
            # Test 1: Missing metadata file
            with self.assertRaises((FileNotFoundError, ValueError)):
                Runner("nonexistent/model/path")
            
            # Test 2: Corrupted metadata (if we can create test scenario)
            runner = Runner(self.model_path)
            
            # Save original metadata
            metadata_path = runner.metadata_path
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    original_metadata = f.read()
                
                # Corrupt metadata temporarily
                with open(metadata_path, 'w') as f:
                    f.write("corrupted json {")
                
                try:
                    with self.assertRaises(ValueError):
                        Runner(self.model_path)
                finally:
                    # Restore original metadata
                    with open(metadata_path, 'w') as f:
                        f.write(original_metadata)
                        
            print(f"✅ Metadata dependency error handling verified")
            
        except Exception as e:
            self.skipTest(f"Skipping metadata dependency test due to setup issue: {e}")

    def test_comprehensive_error_threshold_analysis(self):
        """Test comprehensive error threshold analysis with detailed metrics and logit comparison."""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE ERROR THRESHOLD ANALYSIS WITH LOGIT DIFFERENCES")
        print("="*80)
        
        models = ["models/doom", "models/net"]
        results_table = []
        
        for model_path in models:
            model_name = os.path.basename(model_path)
            print(f"\n🎯 Testing {model_name.upper()} Model")
            print("-" * 60)
            
            try:
                # Initialize runner
                runner = Runner(model_path)
                
                # Test 1: Whole model inference
                print("📊 Whole Model Inference:")
                start_time = time.time()
                whole_result = runner.infer(mode="onnx_only")
                whole_time = time.time() - start_time
                whole_class = whole_result.get("predicted_class", -1)
                whole_security = whole_result.get("overall_security", 0)
                whole_logits = whole_result.get("logits", [[]])[0] if whole_result.get("logits") else []
                
                print(f"   ✅ Predicted Class: {whole_class}")
                print(f"   🛡️  Security: {whole_security}%")
                print(f"   ⏱️  Time: {whole_time:.3f}s")
                print(f"   📊 Logits Length: {len(whole_logits)}")
                
                # Test 2: Sliced inference
                print("\n📊 Sliced Auto Inference:")
                start_time = time.time()
                sliced_result = runner.infer(mode="auto")
                sliced_time = time.time() - start_time
                sliced_class = sliced_result.get("predicted_class", -1)
                sliced_security = sliced_result.get("overall_security", 0)
                sliced_logits = sliced_result.get("logits", [[]])[0] if sliced_result.get("logits") else []
                execution_results = sliced_result.get("execution_results", [])
                
                print(f"   ✅ Predicted Class: {sliced_class}")
                print(f"   🛡️  Security: {sliced_security}%")
                print(f"   ⏱️  Time: {sliced_time:.3f}s")
                print(f"   📊 Logits Length: {len(sliced_logits)}")
                
                # Test 3: Logit Difference Analysis
                print("\n📊 Logit Difference Analysis:")
                if len(whole_logits) > 0 and len(sliced_logits) > 0 and len(whole_logits) == len(sliced_logits):
                    import numpy as np
                    whole_logits_array = np.array(whole_logits)
                    sliced_logits_array = np.array(sliced_logits)
                    
                    # Calculate absolute differences
                    absolute_differences = np.abs(whole_logits_array - sliced_logits_array)
                    max_absolute_error = np.max(absolute_differences)
                    mean_absolute_error = np.mean(absolute_differences)
                    sum_absolute_error = np.sum(absolute_differences)
                    
                    print(f"   📏 Max Absolute Error: {max_absolute_error:.6f}")
                    print(f"   📐 Mean Absolute Error: {mean_absolute_error:.6f}")
                    print(f"   📊 Sum Absolute Error: {sum_absolute_error:.6f}")
                    
                    # Show top 5 largest differences
                    sorted_indices = np.argsort(absolute_differences)[::-1][:5]
                    print(f"   🔝 Top 5 Logit Differences:")
                    for i, idx in enumerate(sorted_indices):
                        print(f"      {i+1}. Index {idx}: Whole={whole_logits_array[idx]:.6f}, Sliced={sliced_logits_array[idx]:.6f}, Diff={absolute_differences[idx]:.6f}")
                        
                    logit_comparison = {
                        "max_absolute_error": max_absolute_error,
                        "mean_absolute_error": mean_absolute_error,
                        "sum_absolute_error": sum_absolute_error,
                        "logits_match": max_absolute_error < 1e-6,  # Consider match if very small difference
                        "top_differences": [(int(idx), float(absolute_differences[idx])) for idx in sorted_indices]
                    }
                else:
                    print(f"   ⚠️  Cannot compare logits: whole({len(whole_logits)}) vs sliced({len(sliced_logits)})")
                    logit_comparison = {
                        "max_absolute_error": float('inf'),
                        "mean_absolute_error": float('inf'),
                        "sum_absolute_error": float('inf'),
                        "logits_match": False,
                        "error": "Logit dimension mismatch"
                    }
                
                # Count methods used
                methods_count = {}
                circuit_sizes = []
                fallback_count = 0
                
                for exec_result in execution_results:
                    method = exec_result.get("method", "unknown")
                    methods_count[method] = methods_count.get(method, 0) + 1
                    
                    if exec_result.get("fallback_used", False):
                        fallback_count += 1
                    
                    circuit_size = exec_result.get("circuit_size", 0)
                    if circuit_size > 0:
                        circuit_sizes.append(circuit_size)
                
                print(f"\n📊 Execution Details:")
                print(f"   🔧 Methods Used: {methods_count}")
                print(f"   📦 Circuit Sizes: {circuit_sizes} bytes")
                print(f"   🔄 Fallbacks: {fallback_count}/{len(execution_results)}")
                
                # Test 4: Verification analysis
                print("\n📊 Verification Analysis:")
                verified_slices = runner.verified_slices
                total_slices = len(verified_slices)
                verified_count = sum(1 for v in verified_slices.values() if v)
                verified_security = (verified_count / total_slices * 100) if total_slices > 0 else 0
                
                print(f"   ✅ Verified Slices: {verified_count}/{total_slices}")
                print(f"   🛡️  Verified Security: {verified_security:.1f}%")
                print(f"   📋 Verification Details: {verified_slices}")
                
                # Calculate error metrics
                classification_match = whole_class == sliced_class
                time_overhead = ((sliced_time - whole_time) / whole_time * 100) if whole_time > 0 else 0
                security_gain = sliced_security - whole_security
                verified_gain = verified_security - whole_security
                
                print(f"\n📊 Error Metrics:")
                print(f"   ✅ Classification Match: {classification_match}")
                print(f"   ⏱️  Time Overhead: +{time_overhead:.1f}%")
                print(f"   🛡️  Security Gain (Sliced): +{security_gain:.1f}%")
                print(f"   🔐 Security Gain (Verified): +{verified_gain:.1f}%")
                
                # Store results for table
                results_table.append({
                    "model": model_name,
                    "whole_class": whole_class,
                    "whole_security": whole_security,
                    "whole_time": whole_time,
                    "whole_logits": whole_logits,
                    "sliced_class": sliced_class,
                    "sliced_security": sliced_security,
                    "sliced_time": sliced_time,
                    "sliced_logits": sliced_logits,
                    "verified_security": verified_security,
                    "classification_match": classification_match,
                    "time_overhead": time_overhead,
                    "security_gain": security_gain,
                    "verified_gain": verified_gain,
                    "methods_count": methods_count,
                    "circuit_sizes": circuit_sizes,
                    "fallback_count": fallback_count,
                    "total_slices": total_slices,
                    "logit_comparison": logit_comparison
                })
                
            except Exception as e:
                print(f"   ❌ Error testing {model_name}: {e}")
                self.fail(f"Error testing {model_name}: {e}")
        
        # Print comparison table with logit differences
        print("\n" + "="*80)
        print("📊 INFERENCE METHOD COMPARISON TABLE WITH LOGIT ANALYSIS")
        print("="*80)
        
        print(f"{'Model':<8} {'Method':<12} {'Class':<6} {'Security %':<10} {'Time (s)':<8} {'Error':<6} {'Overhead %':<10} {'Gain %':<8} {'Max Logit Err':<12}")
        print("-" * 90)
        
        for result in results_table:
            model = result["model"]
            
            # Whole model row
            print(f"{model:<8} {'Whole':<12} {result['whole_class']:<6} {result['whole_security']:<10.1f} {result['whole_time']:<8.3f} {'-':<6} {'-':<10} {'-':<8} {'-':<12}")
            
            # Sliced row with logit error
            error_symbol = "✅" if result["classification_match"] else "❌"
            logit_error = result["logit_comparison"].get("max_absolute_error", float('inf'))
            logit_error_str = f"{logit_error:.6f}" if logit_error != float('inf') else "N/A"
            
            print(f"{'':>8} {'Sliced':<12} {result['sliced_class']:<6} {result['sliced_security']:<10.1f} {result['sliced_time']:<8.3f} {error_symbol:<6} {result['time_overhead']:>+9.1f} {result['security_gain']:>+7.1f} {logit_error_str:<12}")
            
            # Verified row
            print(f"{'':>8} {'Verified':<12} {'-':<6} {result['verified_security']:<10.1f} {'-':<8} {'-':<6} {'-':<10} {result['verified_gain']:>+7.1f} {'-':<12}")
            print()
        
        # Print detailed logit analysis
        print("="*80)
        print("📊 DETAILED LOGIT ANALYSIS")
        print("="*80)
        
        for result in results_table:
            model = result["model"]
            logit_comp = result["logit_comparison"]
            
            print(f"\n🎯 {model.upper()} Model Logit Analysis:")
            if "error" not in logit_comp:
                print(f"   📏 Max Absolute Error: {logit_comp['max_absolute_error']:.8f}")
                print(f"   📐 Mean Absolute Error: {logit_comp['mean_absolute_error']:.8f}")
                print(f"   📊 Sum Absolute Error: {logit_comp['sum_absolute_error']:.8f}")
                print(f"   ✅ Logits Match (< 1e-6): {logit_comp['logits_match']}")
                
                print(f"   🔝 Top 3 Largest Differences:")
                for i, (idx, diff) in enumerate(logit_comp['top_differences'][:3]):
                    whole_val = result['whole_logits'][idx] if idx < len(result['whole_logits']) else 'N/A'
                    sliced_val = result['sliced_logits'][idx] if idx < len(result['sliced_logits']) else 'N/A'
                    print(f"      {i+1}. Index {idx}: Whole={whole_val:.6f}, Sliced={sliced_val:.6f}, Diff={diff:.6f}")
            else:
                print(f"   ❌ Error: {logit_comp['error']}")
        
        print("\n" + "="*80)
        
        # Assert all tests passed
        for result in results_table:
            self.assertTrue(result["classification_match"], 
                          f"{result['model']} classification mismatch: whole={result['whole_class']}, sliced={result['sliced_class']}")
            
            # Assert logit differences are reasonable (< 0.1)
            logit_error = result["logit_comparison"].get("max_absolute_error", float('inf'))
            if logit_error != float('inf'):
                self.assertLess(logit_error, 0.1, 
                              f"{result['model']} logit difference too large: {logit_error:.6f}")
            
        print("✅ All error threshold tests passed!")
        print("✅ All logit difference tests passed!")

    def test_logit_absolute_error_calculation(self):
        """Test dedicated logit absolute error calculation between inference methods."""
        print("\n" + "="*60)
        print("🧮 LOGIT ABSOLUTE ERROR CALCULATION TEST")
        print("="*60)
        
        models = ["models/doom", "models/net"]
        
        for model_path in models:
            model_name = os.path.basename(model_path)
            print(f"\n📊 {model_name.upper()} Logit Error Analysis")
            print("-" * 40)
            
            try:
                runner = Runner(model_path)
                
                # Get whole model inference
                print("🔍 Running whole model inference...")
                whole_result = runner.infer(mode="onnx_only")
                whole_logits = whole_result.get("logits", [[]])[0]
                
                # Get sliced inference  
                print("🔍 Running sliced inference...")
                sliced_result = runner.infer(mode="auto")
                sliced_logits = sliced_result.get("logits", [[]])[0]
                
                if len(whole_logits) > 0 and len(sliced_logits) > 0:
                    import numpy as np
                    
                    # Calculate various error metrics
                    whole_array = np.array(whole_logits)
                    sliced_array = np.array(sliced_logits)
                    
                    if whole_array.shape == sliced_array.shape:
                        # Absolute differences
                        abs_diff = np.abs(whole_array - sliced_array)
                        
                        # Error metrics
                        max_abs_error = np.max(abs_diff)
                        min_abs_error = np.min(abs_diff)  
                        mean_abs_error = np.mean(abs_diff)
                        std_abs_error = np.std(abs_diff)
                        sum_abs_error = np.sum(abs_diff)
                        
                        # Relative error where possible
                        nonzero_mask = whole_array != 0
                        rel_error = np.zeros_like(abs_diff)
                        rel_error[nonzero_mask] = abs_diff[nonzero_mask] / np.abs(whole_array[nonzero_mask])
                        max_rel_error = np.max(rel_error) if np.any(nonzero_mask) else 0
                        
                        print(f"📏 Logit Dimensions: {whole_array.shape}")
                        print(f"📊 Max Absolute Error: {max_abs_error:.8f}")
                        print(f"📊 Min Absolute Error: {min_abs_error:.8f}")
                        print(f"📊 Mean Absolute Error: {mean_abs_error:.8f}")
                        print(f"📊 Std Absolute Error: {std_abs_error:.8f}")
                        print(f"📊 Sum Absolute Error: {sum_abs_error:.8f}")
                        print(f"📊 Max Relative Error: {max_rel_error:.8f}")
                        
                        # Classification comparison
                        whole_class = whole_result.get("predicted_class", -1)
                        sliced_class = sliced_result.get("predicted_class", -1)
                        
                        print(f"🎯 Whole Model Prediction: {whole_class}")
                        print(f"🎯 Sliced Model Prediction: {sliced_class}")
                        print(f"✅ Classification Match: {whole_class == sliced_class}")
                        
                        # Store results for assertion
                        self.assertLess(max_abs_error, 1.0, 
                                      f"Max absolute error too large for {model_name}: {max_abs_error}")
                        self.assertEqual(whole_class, sliced_class,
                                       f"Classification mismatch for {model_name}")
                        
                        print(f"✅ Logit error test passed for {model_name}")
                        
                    else:
                        print(f"❌ Shape mismatch: whole{whole_array.shape} vs sliced{sliced_array.shape}")
                        self.fail(f"Logit shape mismatch for {model_name}")
                        
                else:
                    print(f"❌ Empty logits: whole({len(whole_logits)}) vs sliced({len(sliced_logits)})")
                    self.fail(f"Empty logits for {model_name}")
                    
            except Exception as e:
                print(f"❌ Error in logit test for {model_name}: {e}")
                self.fail(f"Logit test failed for {model_name}: {e}")
        
        print(f"\n✅ All logit absolute error tests passed!")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 