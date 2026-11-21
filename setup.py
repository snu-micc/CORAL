from setuptools import find_packages, setup

setup(
    name="coral-pka",
    version="0.1.0",
    description="Comparative Reference-Anchored Learning for molecular property prediction",
    packages=find_packages(include=["coral", "coral.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scipy>=1.7",
        "tqdm>=4.62",
    ],
)
