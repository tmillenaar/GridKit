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

Then go into the folder and the gridkit-py sub-directory

.. code-block:: shell

    cd GridKit/gridkit-py

Now let's create a venv to use.

.. code-block:: shell

    python3 -m venv gridkit_venv

Activate the venv

.. code-block:: shell

    source gridkit_venv/bin/activate

Upgrade the basics: pip, setuptools, wheel

.. code-block:: shell

    python3 -m pip install pip setuptools wheel

.. Note ::
    When running ``pip install .``, it is assumed that ``rustup`` and ``Cargo`` are installed.
    If these are not installed, see the `rust installation guide <https://doc.rust-lang.org/cargo/getting-started/installation.html>`_.
    The build process also assumes build tools are installed such as ``gcc``  and ``make``.
    On Ubuntu, as used for this example, these can be installed using ``sudo apt install build-essential``.

Install the package with the additional 'doc' and 'test' dependencies.
Here the ``-e`` flag will make sure the the local files are used. If this flag is omitted, a copy is installed in the venv
which will be used over the local files and not reflect any changes you make in the repository.

.. code-block:: shell

    python3 -m pip install -e .[doc,test]

Since the python code partially uses functions defined in Rust, we have to compile the rust binary.
This is done automatically when installing the python package. If changes are made in the rust code
(in the ../gridkit-rs directory) the rust binary needs to be re-build for the changes to take effect.
Run the following to compile the Rust binary for use in python:

.. code-block:: shell

    maturin develop --release

The '--release' flag will do some further optimizations.
This is optional when testing. The compilation will take slightly longer but the code in the binary will run faster.
Either way, the binary will get stored in the gridkit-py/gridkit subfolder and will be called something like
'gridkit_rs.cpython-310-x86_64-linux-gnu.so' or your platform equivalent.


Code formatting
---------------
`black <https://pypi.org/project/black/>`_ and `isort <https://pypi.org/project/isort/>`_ are used for code formatting.
These packages are installed when the ``[test]`` argument is used during installation (see the command above).
Pytest-black will test the format of the python files.
Code that does not pass the test should be reformatted using black

.. code-block:: shell

    python3 -m black gridkit tests/test_gridkit/

and isort

.. code-block:: shell

    python3 -m isort gridkit tests/test_gridkit/

It is recommended to install the pre-commit hook, which will check the code format on commit and fix it if needed

.. code-block:: shell

    pre-commit install

This process should take away a lot of strain around neatly formatting the code and
ensures the same code standards are enforced all over the codebase.

Running tests and building docs
-------------------------------

Now the package is installed, the unittests can be ran by calling pytest

.. code-block:: shell

    python3 -m pytest tests

Run the doctests

.. code-block:: shell

    python3 -m pytest gridkit/ --doctest-modules

To build the documentation, a script was created at docs/build_docs/sh.
This script can be called like so `bash docs/build_docs.sh`, note that in order for the script to work you need to use `bash` and not `sh`.
The script will build documentation for all released versions (after v0.7.0). The script does need to know the
latest version to place outdated warnings on all older versions. To do this the GRIDKIT_DOC_BUILD_LATEST_VERSION environment variable needs to be set.
So you can build all docs by:

.. code-block:: shell

    export GRIDKIT_DOC_BUILD_LATEST_VERSION="v0.14.1"
    bash docs/build_docs.sh

Note here that we arbitrarily set v0.14.1 as the latest release, feel free to change this to the version that is acutally the last release at your time of building the docs.

Building the docs for all versions takes a lot of time though, so if you want to iterate quickly it is recommended to do the following:

.. code-block:: shell

    export GRIDKIT_DOC_BUILD_LATEST_VERSION="dev"
    export GRIDKIT_DOC_BUILD_CURRENT_VERSION="dev"
    export GRIDKIT_DOC_BUILD_TAGS="['dev']"
    python3 -m sphinx.cmd.build docs/source build/sphinx/dev/html

This assumes that you have an environment in which gridkit is installed with the doc dependencies. To install this from a checkout of the repository, run `pip install -e ./[doc]`.

The docs are then stored locally in ``./build/sphinx/dev/html/``.
If you build all the docs using the docs/build_docs.sh script the docs are then stored in ``./build/sphinx/html/``

.. Note ::

    Depending on how you installed sphinx, you might also be able to call `sphinx-build` instead of `python3 -m sphinx.cmd.build`.
    Use whatever you like.

..
