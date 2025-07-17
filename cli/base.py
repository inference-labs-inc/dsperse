"""
Base module for Kubz CLI functionality.
Contains common utilities and classes used by all CLI commands.
"""

import argparse
import random
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Easter eggs
EASTER_EGGS = [
    "Did you know? Neural networks are just spicy linear algebra!",
    "Fun fact: The first neural network was created in 1943 by Warren McCulloch and Walter Pitts.",
    "Pro tip: Always normalize your inputs!",
    "Kubz fact: Slicing models helps with interpretability and verification.",
    "ZK fact: Zero-knowledge proofs allow you to prove you know something without revealing what it is.",
    "Kubz was named after the idea of 'cubes' of computation in neural networks.",
    "The answer to life, the universe, and everything is... 42 (but you need a neural network to understand why).",
    "Neural networks don't actually think. They just do math really fast.",
    "If you're reading this, you're awesome! Keep up the great work!",
    "Kubz: Making neural networks more transparent, one slice at a time."
]

def print_header():
    """Print the Kubz CLI header with ASCII art."""
    header = f"""
{Fore.CYAN}
 ██ ▄█▀ █    ██  ▄▄▄▄   ▒███████▒
 ██▄█▒  ██  ▓██▒▓█████▄ ▒ ▒ ▒ ▄▀░
▓███▄░ ▓██  ▒██░▒██▒ ▄██░ ▒ ▄▀▒░ 
▓██ █▄ ▓▓█  ░██░▒██░█▀  ░ ▄▀▒   ░
▒██▒ █▄▒▒█████▓ ░▓█  ▀█▓▒███████▒
▒ ▒▒ ▓▒░▒▓▒ ▒ ▒ ░▒▓███▀▒░▒▒ ▓░▒░▒
░ ░▒ ▒░░░▒░ ░ ░ ▒░▒   ░ ░░▒ ▒ ░ ▒
░ ░░ ░  ░░░ ░ ░  ░    ░ ░ ░ ░ ░ ░
░  ░      ░      ░      ░ ░    ░ 
                       ░        
{Style.RESET_ALL}
{Fore.YELLOW}Distributed zkML Toolkit{Style.RESET_ALL}
"""
    print(header)

def print_easter_egg():
    """Print a random easter egg."""
    print(f"\n{Fore.GREEN}🥚 {random.choice(EASTER_EGGS)}{Style.RESET_ALL}\n")

# Custom ArgumentParser that shows header and easter egg with help
class KubzArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        print_header()
        if random.random() < 0.2:  # 20% chance to show an easter egg
            print_easter_egg()
        super().print_help(file)

    def add_subparsers(self, **kwargs):
        # Ensure that subparsers are also KubzArgumentParser instances
        kwargs.setdefault('parser_class', KubzArgumentParser)
        return super().add_subparsers(**kwargs)

def check_model_dir(model_dir):
    """
    Check if the model directory exists.

    Args:
        model_dir (str): Path to the model directory

    Returns:
        bool: True if the directory exists, False otherwise
    """
    import os
    if not os.path.exists(model_dir):
        print(f"{Fore.RED}Error: Model directory '{model_dir}' does not exist.{Style.RESET_ALL}")
        return False
    return True

def detect_model_type(model_dir):
    """
    Detect the model type (ONNX or PyTorch) based on the files in the model directory.

    Args:
        model_dir (str): Path to the model directory

    Returns:
        tuple: (is_onnx, error_message) where is_onnx is a boolean and error_message is a string or None
    """
    import os
    is_onnx = False
    error_message = None

    if os.path.exists(os.path.join(model_dir, "model.onnx")):
        is_onnx = True
    elif not os.path.exists(os.path.join(model_dir, "model.pth")):
        error_message = f"{Fore.RED}Error: No model.pth or model.onnx found in '{model_dir}'.{Style.RESET_ALL}"

    return is_onnx, error_message

def save_result(result, output_file):
    """
    Save the result to a file.

    Args:
        result: The result to save
        output_file (str): Path to the output file
    """
    import json
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"{Fore.GREEN}✓ Results saved to {output_file}{Style.RESET_ALL}")

def prompt_for_value(param_name, prompt_message, default=None, required=True):
    """
    Prompt the user for a value if it's missing.

    Args:
        param_name (str): The name of the parameter
        prompt_message (str): The message to display when prompting
        default (str, optional): The default value to use if the user just presses enter
        required (bool, optional): Whether the parameter is required

    Returns:
        str: The value provided by the user or the default value
    """
    try:
        if default:
            user_input = input(f"{Fore.YELLOW}{prompt_message} [{default}]: {Style.RESET_ALL}")
            if not user_input.strip():
                return default
            return user_input.strip()
        else:
            while True:
                user_input = input(f"{Fore.YELLOW}{prompt_message}: {Style.RESET_ALL}")
                if user_input.strip() or not required:
                    return user_input.strip()
                print(f"{Fore.RED}Error: {param_name} is required.{Style.RESET_ALL}")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Error getting input: {e}{Style.RESET_ALL}")
        return default if not required else None
