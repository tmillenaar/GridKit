from setuptools import setup, find_packages

with open("version.py") as file:
    version_info = dict(line.replace(" ","").split("=") for line in file)

with open("requirements.txt") as file:
    install_requires = file.read()

tests_require = [
    "astroid",
    "black",
    "coverage>=4",
    "flake8",
    "mypy",
    "pytest>=3.0",
    "pytest-forked",
    "pytest-cov>=2.3.1",
    "pytest-pylint",
    "pytest-black",
    "pylint",
    "pre-commit",
]

docs_require = [
    "sphinx!=3.5.0",
    "sphinx_rtd_theme",
    "sphinxcontrib-plantuml",
]

setup(
    name="grid",
    version=version_info["__version__"],
    author="Timo Millenaar",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[install_requires],
    tests_require=tests_require,
    extras_require={"test": tests_require, "doc": docs_require},
)