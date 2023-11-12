from pathlib import Path

from setuptools import find_packages, setup

with open("gridkit/version.py") as file:
    version_info = dict(
        line.replace(" ", "").replace('"', "").split("=") for line in file
    )

with open("requirements.txt") as file:
    install_requires = file.read()

docs_require = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-gallery",
    "sphinxcontrib-plantuml",
    "dask_geopandas",
    "matplotlib",
]

tests_require = [
    "pytest",
    "black",
    "geopandas",
    "maturin",
    "pytest-cov",
    "pytest-pylint",
    "pytest-black",
    "pylint",
    "pre-commit",
    *docs_require,
]

setup(
    name="gridkit",
    version=version_info["__version__"],
    author="Timo Millenaar",
    description="Powerful abstractions of infinite grids for grid-vector interactions, tesselation, resampling and interactions between related grids.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[install_requires],
    tests_require=tests_require,
    extras_require={"test": tests_require, "doc": docs_require},
)
