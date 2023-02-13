from setuptools import setup, find_packages

with open("gridkit/version.py") as file:
    version_info = dict(line.replace(" ","").replace('"', "").split("=") for line in file)

with open("requirements.txt") as file:
    install_requires = file.read()

tests_require = [
    "black",
    "geopandas",
    "pytest-pylint",
    "pytest-black",
    "pylint",
    "pre-commit",
]

docs_require = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-gallery",
    "sphinxcontrib-plantuml",
]

setup(
    name="gridkit",
    version=version_info["__version__"],
    author="Timo Millenaar",
    description="Operations on, and tessellation of, regular grids",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[install_requires],
    tests_require=tests_require,
    extras_require={"test": tests_require, "doc": docs_require},
)