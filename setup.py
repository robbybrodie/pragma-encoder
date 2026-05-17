"""Setup configuration for pragma-encoder.

An independent open-source implementation of the PRAGMA foundation model
architecture from Ostroukhov et al. (2026), arXiv:2604.08649v1.

Install in development mode:
    pip install -e .

This is not affiliated with Revolut or NVIDIA. No pretrained weights
or proprietary code are included.
"""

from setuptools import setup, find_packages

setup(
    name="pragma-encoder",
    version="0.1.0",
    description=(
        "Open source implementation of the PRAGMA foundation model "
        "architecture (Ostroukhov et al., 2026, arXiv:2604.08649v1)"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="pragma-encoder contributors",
    license="Apache-2.0",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "einops>=0.7.0",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.19.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "umap-learn>=0.5.0",
        ],
        "pipeline": [
            "kfp>=2.0.0",
            "kfp-kubernetes>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
