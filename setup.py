from setuptools import setup, find_packages
import os

# Get the directory where setup.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))
# Read requirements from the requirements.txt file, handling the case if it doesn't exist
requirements = []
requirements_path = os.path.join(base_dir, 'requirements.txt')
if os.path.exists(requirements_path):
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="dsperse",
    version="1.0.0",
    description="Distributed zkML Toolkit",
    author="Inference Labs",
    author_email="info@inferencelabs.com",
    packages=find_packages(),
    py_modules=["main"],  # Include main.py in the package
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dsperse=main:main",
        ],
    },
)
