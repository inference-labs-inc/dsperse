from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="kubz",
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
            "kubz=main:main",
        ],
    },
)
