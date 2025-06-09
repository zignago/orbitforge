from setuptools import setup, find_packages

setup(
    name="orbitforge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer>=0.9.0",
        "pydantic>=2.0.0",
        "pytest>=7.0.0",
        "rich>=10.0.0",
        "pyyaml>=6.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "gmsh>=4.11.0",
        "loguru>=0.6.0",
    ],
    entry_points={
        "console_scripts": [
            "orbitforge=orbitforge.cli:app",
        ],
    },
    python_requires=">=3.8",
    author="OrbitForge Team",
    description="Generate CubeSat structures from mission specs",
    long_description=open("docs/Orbitforge_Project_Context.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.8",
    ],
)
