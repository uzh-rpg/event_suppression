from setuptools import find_packages, setup


setup(
    name="evsup",
    version="0.1.0",
    author="Roberto Pellerito",
    description="Training and validation code for Event Suppressor.",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24",
        "h5py>=3.6",
        "hdf5plugin>=3.2",
        "opencv-python>=4.5",
        "scipy>=1.10",
        "tqdm>=4.64",
        "PyYAML>=6.0",
    ],
)
