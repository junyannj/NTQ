from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ntq",
    version="0.1.0",
    author="Kizamuel",
    author_email="your.email@example.com",
    description="Negative thermal quenching analysis through temperature-dependent PL/EL fitting using Marcus-Levich-Jortner formalism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/junyannj/NTQ",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "emcee",
        "pandas",
        "h5py",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "notebook": [
            "ipykernel",
            "jupyter",
            "openpyxl",
        ],
    },
)