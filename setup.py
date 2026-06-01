from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        return [
            line for line in lines
            if line and not line.startswith(('#', '-', '--'))
        ]


setup(
    name="dynamic_masker",
    version="0.1.0",
    author="Roberto Pellerito",
    author_email="rpellerito@ifi.uzh.ch",
    description="A package for dynamic object masking in visual odometry.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/senecobis/dynamic_masker.git",  # Update if hosted
    packages=find_packages(),  # Automatically finds subpackages
    # install_requires=parse_requirements('requirements.txt'),
    install_requires=[
        "torch>=1.10.0", 
        "numpy>=1.21.0", 
        "matplotlib>=3.4.0",
        # Add other dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
