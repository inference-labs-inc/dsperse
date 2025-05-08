import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Set

import psutil
import torch
import torch.nn.functional as F

from src.utils.model_utils import ModelUtils


def get_mem_usage_cli(pid: int) -> int:
    """Get memory usage (resident set size in KB) from ps."""
    if sys.platform != 'darwin':
        # ps -o rss= might work on Linux, but vmmap won't
        return 0
    try:
        # Use full path to be sure we get the system's ps
        command = ['/bin/ps', '-p', str(pid), '-o', 'rss=']
        output = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        # Output might be empty if process died; strip whitespace before int conversion
        rss_kb_str = output.decode().strip()
        return int(rss_kb_str) if rss_kb_str else 0
    except subprocess.CalledProcessError:
        # Process likely doesn't exist anymore
        return 0

    except Exception as e:
        print(f"Error getting RSS via CLI for PID {pid}: {e}")
        return 0


def get_swap_usage_cli(pid: int) -> int:
    """Estimate swap usage (in KB) using vmmap. Supports both legacy and modern vmmap output."""
    if sys.platform != 'darwin':
        return 0
    try:
        command = ['/usr/bin/sudo', '/usr/bin/vmmap', str(pid)]

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=5)

        if process.returncode != 0:
            # print(f"[Debug Swap PID {pid}] vmmap exited with code {process.returncode}")
            if stderr:
                pass
                # print(f"[Debug Swap PID {pid}] vmmap stderr:\n{stderr.strip()}")
            return 0

        output = stdout

        # First attempt: legacy "Swap used"
        match_old = re.search(r'Swap used:\s+([\d,]+)\s*(K|KB)', output, re.IGNORECASE)
        if match_old:
            val = int(match_old.group(1).replace(',', ''))
            return val

        # Second attempt: modern "swapped_out=" or "swapped_out ="
        match_new = re.search(r'swapped[_ ]?out\s*=\s*([\d.]+)\s*([KMG])', output, re.IGNORECASE)
        if match_new:
            val = float(match_new.group(1))
            unit = match_new.group(2).upper()
            multiplier = {'K': 1, 'M': 1024, 'G': 1024 * 1024}.get(unit, 1)
            swap_kb = int(val * multiplier)
            return swap_kb

        return 0

    except Exception as e:
        print(f"Error getting Swap via vmmap for PID {pid}: {e}")
        return 0

def monitor_subprocess_memory(parent_pid, process_name_keyword, results, stop_event):
    """
    Monitors child processes of parent_pid using CLI tools (ps, vmmap).
    Tracks the peak SUM of RSS ('mem_cli') and Swap ('swap_cli') across
    all children matching the keyword.
    Updates the 'results' dictionary.
    """
    peak_mem_cli = 0
    peak_swap_cli = 0
    tracked_pids: Set[int] = set()

    # Ensure the results dictionary exists and initialize keys
    if not isinstance(results, dict):
        return

    # Use different keys to distinguish from psutil results
    results['peak_subprocess_mem'] = 0
    results['peak_subprocess_swap'] = 0
    results['peak_subprocess_total'] = 0 # Mem + Swap

    try:
        parent = psutil.Process(parent_pid)

        while not stop_event.is_set():
            current_total_mem_cli = 0
            current_total_swap_cli = 0
            children_found_this_cycle = []
            try:
                # Still use psutil to reliably get all descendants
                children_found_this_cycle = parent.children(recursive=True)
            except psutil.NoSuchProcess:
                # print(f"Parent process {parent_pid} seems to have ended. Stopping CLI monitor.")
                break
            except Exception as e:
                # print(f"Error getting children for PID {parent_pid}: {e}")
                time.sleep(0.1) # Avoid tight loop on error
                continue

            active_pids_this_cycle = set()

            # First pass: Identify and add new matching children
            for proc in children_found_this_cycle:
                try:
                    pid = proc.pid
                    active_pids_this_cycle.add(pid) # Keep track of currently running children
                    if pid not in tracked_pids:
                         # Check name only for newly found processes
                        proc_name = proc.name()
                        if process_name_keyword.lower() in proc_name.lower():
                            tracked_pids.add(pid)
                            # print(f"CLI Monitoring child: {proc_name} (PID: {pid})")

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process ended or permissions issue while checking name/pid
                    if pid in tracked_pids:
                        tracked_pids.remove(pid) # Stop tracking
                    continue
                except Exception as e:
                    # print(f"Error inspecting potential child {pid}: {e}")
                    if pid in tracked_pids:
                         tracked_pids.remove(pid) # Stop tracking
                    continue

            # Second pass: Check memory for all currently tracked PIDs
            pids_to_remove = set()
            for pid in tracked_pids:
                if pid not in active_pids_this_cycle:
                     # Process we were tracking is no longer a child (finished?)
                     pids_to_remove.add(pid)
                     continue # Don't try to get memory for it

                # Get memory and swap using CLI tools
                mem_kb = get_mem_usage_cli(pid)
                swap_kb = get_swap_usage_cli(pid)

                # Add to the total for this cycle
                # Note: If get_mem/swap returns 0 (e.g., process just died), it adds 0
                current_total_mem_cli += mem_kb
                current_total_swap_cli += swap_kb

            # Remove PIDs that are no longer active
            tracked_pids.difference_update(pids_to_remove)
            if pids_to_remove:
                pass
                 # print(f"Stopped CLI tracking for finished PIDs: {pids_to_remove}")


            # Update overall peaks for the SUM of memory/swap
            peak_mem_cli = max(peak_mem_cli, current_total_mem_cli)
            peak_swap_cli = max(peak_swap_cli, current_total_swap_cli)

            # Break if stop signal received after checking all children this cycle
            if stop_event.is_set():
                break

            # Sleep before next cycle (might need longer sleep due to CLI overhead)
            time.sleep(0.1) # Increased sleep interval

    except psutil.NoSuchProcess:
        print(f"Initial parent process {parent_pid} not found.")
    except Exception as e:
        print(f"Major error in CLI monitoring thread: {e}")
    finally:
        # Store final peak results (convert KB to Bytes for consistency if desired)
        # Store in KB as originally retrieved
        results['peak_subprocess_mem'] = peak_mem_cli
        results['peak_subprocess_swap'] = peak_swap_cli
        results['peak_subprocess_total'] = peak_mem_cli + peak_swap_cli



class RunnerUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def preprocess_input(input_path:str, model_directory: str = None, save_reshape: bool = False) -> torch.Tensor:
        """
        Preprocess input data from JSON.
        """
        try:
            with open(input_path, 'r') as f:
                input_data = json.load(f)

            if isinstance(input_data, dict):
                if 'input_data' in input_data:
                    input_data = input_data['input_data']
                elif 'input' in input_data:
                    input_data = input_data['input']

            # Convert to tensor
            if isinstance(input_data, list):
                if isinstance(input_data[0], list):
                    # 2D input
                    input_tensor = torch.tensor(input_data, dtype=torch.float32)
                else:
                    # 1D input
                    input_tensor = torch.tensor([input_data], dtype=torch.float32)
            else:
                raise ValueError("Expected input data to be a list or nested list")

            # reshape input tensor for the model
            input_tensor = RunnerUtils.reshape(input_tensor, model_directory=model_directory)
            if save_reshape:
                ModelUtils.save_tensor_to_json(input_tensor, "input_data_reshaped.json", model_directory)
                
            return input_tensor

        except Exception as e:
            raise Exception(f"Error preprocessing input: {e}")
        
        
    @staticmethod
    def process_final_output(torch_tensor):
        """Process the final output of the model."""
        # Apply softmax to get probabilities if not already applied
        if len(torch_tensor.shape) != 2:  # Ensure raw output is 2D [batch_size, num_classes]
            print(f"Warning: Raw output shape {torch_tensor.shape} is not as expected. Reshaping to [1, -1].")
            torch_tensor = torch_tensor.reshape(1, -1)

        probabilities = F.softmax(torch_tensor, dim=1)
        predicted_action = torch.argmax(probabilities, dim=1).item()

        result = {
            "logits": torch_tensor,
            "probabilities": probabilities,
            "predicted_action": predicted_action
        }

        return result

    @staticmethod
    def needs_reshape(input_tensor, model_directory: str = None) -> bool:
        """Check if tensor needs reshaping based on dimensions and model type"""
        if input_tensor.dim() != 2:
            return False
        if input_tensor.size(1) == 3136 and model_directory and 'doom' in model_directory.lower():
            return True
        if input_tensor.size(1) == 3072:
            return True
        return False

    @staticmethod
    def reshape(input_tensor, model_directory: str = None):
        # TODO: remove hardcoding to doom and net
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.dim() == 2 and input_tensor.size(1) == 3136 and 'doom' in model_directory.lower():
            input_tensor = input_tensor.reshape(1, 4, 28, 28)
        elif input_tensor.dim() == 2 and input_tensor.size(1) == 3072:
            if input_tensor.size(0) == 1:
                # Single sample case
                input_tensor = input_tensor.reshape(1, 3, 32, 32)
            else:
                # Multiple samples case (e.g., batch size 100)
                print(f"Processing only the first sample out of {input_tensor.size(0)}")
                input_tensor = input_tensor[0:1].reshape(1, 3, 32, 32)
        else:
            raise ValueError(f"Input tensor has unsupported dimensions: {input_tensor.shape}")

        return input_tensor

    @staticmethod
    def get_segments(slices_directory):
        metadata = ModelUtils.load_metadata(slices_directory)
        if metadata is None:
            return None

        segments = metadata.get('segments', [])
        if not segments:
            print("No segments found in metadata.json")
            return None

        num_segments = len(segments)
        return segments

    @staticmethod
    def save_to_file_shaped(input_tensor: torch.Tensor, file_path: str):
        # Convert tensor to list
        tensor_data = input_tensor.tolist()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save tensor data as JSON
        data = {
            "input": tensor_data
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def save_to_file_flattened(input_tensor: torch.Tensor, file_path: str):
        # Flatten and convert tensor to list
        tensor_data = input_tensor.flatten().tolist()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save flattened tensor data as JSON
        data = {
            "input_data": [tensor_data]
        }

        with open(file_path, 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    pass