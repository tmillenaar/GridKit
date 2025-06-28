.. _contributing:

Contributing
============

Currently, GridKit is a solo maintained project, but **help is most welcome**!

Since this is a small project there is no forum or mailing list, but please feel free to
start a `GitHub issue <https://github.com/tmillenaar/GridKit/issues>`_, pull request or contact me directly
on tmillenaar -at- gmail -dot- com.
I'd love to get to know anyone interested in the package.
I'd also love to learn what others might create with GridKit.

Installing for development
--------------------------

Naturally, you'll first want to fork or clone the repository.

.. Note ::
    There are of course many ways of deploying software.
    In this example I will create a new virtual environment using Python's 'venv' package on an Ubuntu machine

Of course we start by cloning or forking from `GitHub <https://github.com/tmillenaar/GridKit>`_.

Then go into the folder
``cd GridKit``
Now let's create a venv to use.

``python3 -m venv gridkit_venv``

Activate the venv

``source gridkit_venv/bin/activate``

Upgrade the basics: pip, setuptools, wheel

``python3 -m pip install pip setuptools wheel``

Install the package with the additional 'doc' and 'test' dependencies.
Here the ``-e`` flag will prevent GridKit itself from being installed in the venv and should
create an 'easy-install.pth' file that points to this local repository.

``python3 -m pip install -e .[doc,test]``

Since the python code partially uses functions defined in Rust, we first have to compile the rust binary.
Run the following to compile the Rust binary:

``maturin develop --release``

The '--release' flag will do some further optimizations.
This is optional when testing. The compilation will take slightly longer but the code in the binary will run faster.
Either way, the binary will get stored in the ./gridkit subfolder and will be called 
'gridkit_rs.cpython-310-x86_64-linux-gnu.so' or your platform equivalent.


Code formatting
---------------
`black <https://pypi.org/project/black/>`_ and `isort <https://pypi.org/project/isort/>`_ are used for code formatting.
These packages are installed when the ``[test]`` argument is used during installation (see the command above).
Pytest-black will test the format of the python files.
Code that does not pass the test should be reformatted using black

``python3 -m black gridkit tests/test_gridkit/``

and isort

``python3 -m isort gridkit tests/test_gridkit/``

It is recommended to install the pre-commit hook, which will check the code format on commit and fix it if needed

``pre-commit install``

This process should take away a lot of strain around neatly formatting the code and
ensures the same code standards are enforced all over the codebase.

Running tests and building docs
-------------------------------

Now the package is installed, the unittests can be ran by calling pytest

``python3 -m pytest tests``

Run the doctests

``python3 -m pytest gridkit/ --doctest-modules``

And build the documentation locally using Sphinx

``build-sphinx docs/source build/sphinx/html``

The docs are then stored locally in ``./build/sphinx/html/``


