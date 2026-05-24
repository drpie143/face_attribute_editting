from setuptools import setup, find_packages

setup(
    name="face-attr-edit",
    version="1.0.0",
    description="Face Attribute Editing via LoRA-Finetuned Diffusion Models",
    author="Khoa Ngo",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=2.0,<2.6",
        "Pillow>=10.4,<12",
        "opencv-python-headless>=4.12,<5",
        "matplotlib>=3.9,<4",
        "pandas>=2.2.3,<2.4",
        "tqdm>=4.67,<5",
        "scikit-learn>=1.6,<2",
        "PyYAML>=6.0.3,<6.1",
        "streamlit>=1.50,<2",
        "diffusers==0.31.0",
        "transformers>=4.46,<5",
        "accelerate>=1.1,<2",
        "peft>=0.13,<1",
        "safetensors>=0.4.5,<1",
        "lpips==0.1.4",
    ],
)
