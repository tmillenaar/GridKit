# Copyright 2021, SkyGeo Netherlands B.V.

import datetime
import os


def scan_dir(module_dir, parents):

    assert os.path.isdir(module_dir)
    if not os.path.exists(os.path.join(module_dir, "__init__.py")):
        return

    module = parents + (os.path.basename(module_dir),)
    yield ".".join(module)

    # find submodules
    for name in os.listdir(module_dir):
        if name == "__init__.py":
            pass
        elif name.endswith(".py"):
            yield ".".join(module + (name[:-3],))
        elif os.path.isdir(os.path.join(module_dir, name)):
            yield from scan_dir(os.path.join(module_dir, name), module)


def gen_module_rst(module, all_modules, output_dir):

    submodules = tuple(
        name
        for name in all_modules
        if name.startswith(module + ".") and len(name.split(".")) == len(module.split(".")) + 1
    )

    with open(os.path.join(output_dir, module + ".rst"), "w") as rst:
        p_rst = lambda *args, **kwargs: print(*args, file=rst, **kwargs)
        p_rst()
        p_rst(module.split(".")[-1])
        p_rst("=" * len(module))
        p_rst()

        if submodules:
            p_rst("Submodules:")
            p_rst()
            p_rst(".. toctree::")
            p_rst("   :maxdepth: 1")
            p_rst()
            for name in submodules:
                p_rst("   {}".format(name))
            p_rst()

        p_rst(".. automodule:: {}".format(module))
        p_rst("    :members:")
        p_rst("    :undoc-members:")
        p_rst("    :special-members:")
        p_rst("    :show-inheritance:")


def gen_api_reference(module_dir, output_dir):

    modules = tuple(sorted(scan_dir(module_dir, ())))
    for module in modules:
        gen_module_rst(module, modules, output_dir)


# vim: sts=4:sw=4:et
