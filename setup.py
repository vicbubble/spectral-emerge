from setuptools import setup, find_packages

setup(
    name="spectral_emerge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchdiffeq>=0.2.3",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.3",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "pytest>=7.4.0",
        "einops>=0.7.0"
    ],
)
