from setuptools import setup,find_packages

setup(
    name="minitorch",
    version="1.0",
    description=" Mini Pytorch from scratch for deep learning internals",
    author="Adan Nvir",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
    ],
)

