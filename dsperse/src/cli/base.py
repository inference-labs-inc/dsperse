"""
Base module for DSperse CLI functionality.
Contains common utilities and classes used by all CLI commands.
"""

import argparse
import random
import colorama
import logging
import os
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Configure logging
logger = logging.getLogger('dsperse')

# --- Path normalization helpers ---

def normalize_path(p: str) -> str:
    """
    Expand ~ and environment variables, then return an absolute, normalized path.
    Returns the input unchanged if falsy.
    """
    if not p:
        return p
    try:
        expanded = os.path.expanduser(os.path.expandvars(str(p)))
        return os.path.normpath(os.path.abspath(expanded))
    except Exception:
        return p


def _param_name_suggests_path(name: str) -> bool:
    if not name:
        return False
    name = name.lower()
    for token in ("path", "dir", "file", "model", "slices", "output", "input", "run"):
        if token in name:
            return True
    return False


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    v = str(value)
    if v.startswith("~"):
        return True
    if v.startswith("/") or v.startswith("./") or v.startswith("../"):
        return True
    if os.sep in v or (os.altsep and os.altsep in v):
        return True
    if v.lower().endswith((".json", ".onnx", ".pth")):
        return True
    return False


def _maybe_normalize_from_prompt(param_name: str, prompt_message: str, value: str) -> str:
    try:
        if _param_name_suggests_path(param_name) or _looks_like_path(value) or (
            prompt_message and any(t in prompt_message.lower() for t in ["path", "directory", "dir", "file"])  # heuristic
        ):
            return normalize_path(value)
    except Exception:
        pass
    return value


def configure_logging(log_level='WARNING'):
    """
    Configure the logging system.

    Args:
        log_level (str): The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    # Only call basicConfig if logging hasn't been configured yet
    if not logging.root.handlers:
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # If basicConfig was already called, just update the level
        logging.root.setLevel(numeric_level)

    # Ensure all existing loggers respect the new level
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(numeric_level)

    # Set the level for the dsperse logger specifically
    logger.setLevel(numeric_level)

    # Ensure future loggers also respect this level by setting a default level
    logging.getLogger().setLevel(numeric_level)

# Easter eggs
EASTER_EGGS = [
    "Did you know? Neural networks are just spicy linear algebra!",
    "Fun fact: The first neural network was created in 1943 by Warren McCulloch and Walter Pitts.",
    "Pro tip: Always normalize your inputs!",
    "DSperse fact: Slicing models helps with interpretability and verification.",
    "ZK fact: Zero-knowledge proofs allow you to prove you know something without revealing what it is.",
    "DSperse makes it easier to reason about model segments.",
    "The answer to life, the universe, and everything is... 42 (but you need a neural network to understand why).",
    "Neural networks don't actually think. They just do math really fast.",
    "If you're reading this, you're awesome! Keep up the great work!",
    "DSperse: Making neural networks more transparent, one slice at a time."
]

def print_header():
    """Print the DSperse CLI header with ASCII art."""
    header = f"""
{Fore.CYAN}
8888888b.   .d8888b.                                              
888  "Y88b d88P  Y88b                                             
888    888 Y88b.                                                 
888    888  "Y888b.   88888b.   .d88b.  888d888 .d8888b   .d88b.  
888    888     "Y88b. 888 "88b d8P  Y8b 888P"   88K      d8P  Y8b 
888    888       "888 888  888 88888888 888     "Y8888b. 88888888 
888  .d88P Y88b  d88P 888 d88P Y8b.     888          X88 Y8b.     
8888888P"   "Y8888P"  88888P"   "Y8888  888      88888P'  "Y8888  
                      888                                          
                      888                                          
                      888                                          
{Style.RESET_ALL}
{Fore.YELLOW}Distributed zkML Toolkit{Style.RESET_ALL}
"""
    print(header)  # Keep print for header as it's visual UI element
    logger.info("DSperse CLI started")

def print_easter_egg():
    """Print a random easter egg."""
    easter_egg = f"\n{Fore.GREEN}🥚 {random.choice(EASTER_EGGS)}{Style.RESET_ALL}\n"
    print(easter_egg)  # Keep print for easter egg as it's visual UI element
    logger.debug(f"Easter egg displayed: {random.choice(EASTER_EGGS)}")

# Custom ArgumentParser that shows header and easter egg with help
class DsperseArgumentParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        # print_header()
        if random.random() < 0.2:  # 20% chance to show an easter egg
            print_easter_egg()
        super().print_help(file)

    def add_subparsers(self, **kwargs):
        # Ensure that subparsers are also DsperseArgumentParser instances
        kwargs.setdefault('parser_class', DsperseArgumentParser)
        return super().add_subparsers(**kwargs)

def check_model_dir(model_dir):
    """
    Check if the model directory or file exists.

    Args:
        model_dir (str): Path to the model directory or file

    Returns:
        bool: True if the path exists, False otherwise
    """
    normalized = normalize_path(model_dir)
    if not os.path.exists(normalized):
        error_msg = f"Path '{model_dir}' does not exist."
        print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        return False
    logger.debug(f"Model directory/file exists: {normalized}")
    return True

def detect_model_type(model_path):
    """
    Detect the model type (ONNX or PyTorch) based on the file path or files in the directory.

    Args:
        model_path (str): Path to the model file or directory

    Returns:
        tuple: (is_onnx, error_message) where is_onnx is a boolean and error_message is a string or None
    """
    import os
    is_onnx = False
    error_message = None

    # Check if the path is a file
    if os.path.isfile(model_path):
        # Determine model type from file extension
        if model_path.lower().endswith('.onnx'):
            is_onnx = True
            logger.debug(f"Detected ONNX model from file extension: {model_path}")
        elif model_path.lower().endswith('.pth'):
            is_onnx = False
            logger.debug(f"Detected PyTorch model from file extension: {model_path}")
        else:
            error_msg = f"Unsupported model file format. Expected .onnx or .pth file."
            error_message = f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}"
            logger.error(error_msg)
    else:
        # Check for model files in the directory
        if os.path.exists(os.path.join(model_path, "model.onnx")):
            is_onnx = True
            logger.debug(f"Detected ONNX model in directory: {model_path}")
        elif os.path.exists(os.path.join(model_path, "model.pth")):
            is_onnx = False
            logger.debug(f"Detected PyTorch model in directory: {model_path}")
        else:
            error_msg = f"No model.pth or model.onnx found in '{model_path}'."
            error_message = f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}"
            logger.error(error_msg)

    return is_onnx, error_message

def save_result(result, output_file):
    """
    Save the result to a file.

    Args:
        result: The result to save
        output_file (str): Path to the output file
    """
    import json
    from pathlib import Path

    def _default(o):
        # Safely convert non-serializable objects
        if isinstance(o, Path):
            return str(o)
        # Fallback to string representation
        return str(o)

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=_default)
    success_msg = f"Results saved to {output_file}"
    print(f"{Fore.GREEN}✓ {success_msg}{Style.RESET_ALL}")
    logger.info(success_msg)

def prompt_for_value(param_name, prompt_message, default=None, required=True):
    """
    Prompt the user for a value if it's missing. Normalizes path-like inputs.

    Args:
        param_name (str): The name of the parameter
        prompt_message (str): The message to display when prompting
        default (str, optional): The default value to use if the user just presses enter
        required (bool, optional): Whether the parameter is required

    Returns:
        str: The value provided by the user or the default value
    """
    try:
        logger.debug(f"Prompting for {param_name} with message: {prompt_message}")
        if default is not None:
            user_input = input(f"{Fore.YELLOW}{prompt_message} [{default}]: {Style.RESET_ALL}")
            if not user_input.strip():
                # Don't normalize run names (they're not real paths until resolved)
                if str(default).startswith('run_'):
                    logger.debug(f"Using default run name for {param_name}: {default}")
                    return str(default)
                else:
                    normalized_default = _maybe_normalize_from_prompt(param_name, prompt_message, str(default))
                    logger.debug(f"Using default value for {param_name}: {normalized_default}")
                    return normalized_default
            value = user_input.strip().strip('\'"')  # Strip surrounding quotes
            # Don't normalize run names from user input either
            if value.startswith('run_'):
                logger.debug(f"User provided run name for {param_name}: {value}")
                return value
            else:
                value = _maybe_normalize_from_prompt(param_name, prompt_message, value)
                logger.debug(f"User provided value for {param_name}: {value}")
                return value
        else:
            while True:
                user_input = input(f"{Fore.YELLOW}{prompt_message}: {Style.RESET_ALL}")
                if user_input.strip() or not required:
                    if user_input.strip():
                        value = user_input.strip().strip('\'"')  # Strip surrounding quotes
                        value = _maybe_normalize_from_prompt(param_name, prompt_message, value)
                        logger.debug(f"User provided value for {param_name}: {value}")
                        return value
                    else:
                        logger.debug(f"Empty value provided for non-required parameter {param_name}")
                        return user_input.strip()
                error_msg = f"{param_name} is required."
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
                logger.warning(error_msg)
    except KeyboardInterrupt:
        cancel_msg = "Operation cancelled by user."
        print(f"\n{Fore.YELLOW}{cancel_msg}{Style.RESET_ALL}")
        logger.info(cancel_msg)
        import sys
        sys.exit(1)
    except Exception as e:
        error_msg = f"Error getting input: {e}"
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
        logger.error(error_msg)
        return default if not required else None
