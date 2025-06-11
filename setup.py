from setuptools import setup, find_packages

setup(
    name="orbitforge",
    version="0.1.2",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pydantic>=2.0.0",
        "typer>=0.9.0",
        "rich>=13.0.0",
        "loguru>=0.7.0",
        "gmsh>=4.11.0",
        "trimesh>=3.22.0",
        "reportlab>=4.0.0",
        "PyYAML>=6.0.0",
        "pythonocc-core>=7.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "orbitforge=orbitforge.cli:app",
        ],
    },
    package_data={
        "orbitforge": ["generator/materials.yaml"],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Generate flight-ready CubeSat structures from mission specifications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/orbitforge",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
