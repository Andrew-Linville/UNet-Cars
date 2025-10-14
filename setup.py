from setuptools import setup, find_packages

setup(
    name="unet-car",
    version="0.1.0",
    description="U-Net training pipeline for car segmentation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
)
