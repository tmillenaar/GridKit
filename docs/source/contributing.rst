.. _contributing:

Contributing
============

Currently, GridKit is a solo maintained project, but **help is most welcome**!

Since this is a small project there is no forum or mailing list, but please feel free to start a github issue, pull request or contact me directly
on tmillenaar -at- gmail -dot- com.
I'd love to get to know anyone interested in the package.
I'd also love to learn what others might create with GridKit.

Installing for development
--------------------------

Naturally, you'll first want to fork or clone the repository.

.. Note ::
    There are of course many ways of deploying software.
    In this example I will create a new virtual environment using Python's 'venv' package on an Ubuntu machine

I'll assuming you now have a local version of the repository.

First go into the folder
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

Running tests and building docs
-------------------------------

Now the package is installed, the unittests can be ran by calling pytest

``python3 -m pytest tests``

Run the doctests

``python3 -m pytest gridkit/ --doctest-modules``

And build the documentation locally using Sphinx

``python3 setup.py build_sphinx``

The docs are by default stored locally in ``build/sphinx/html/``


