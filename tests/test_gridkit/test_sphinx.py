import os
import subprocess

import pytest


# Define the function to test documentation build
def test_documentation_build(tmp_path):
    build_dir = tmp_path / "_build/html"

    # Run the Sphinx build command and capture its output
    try:
        result = subprocess.run(
            ["sphinx-build", "docs/source", build_dir],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # If the Sphinx build process returns a non-zero exit code, the build failed
        pytest.fail(f"Documentation build failed with error: {e.stderr}")

    assert build_dir.exists()

    expected_build_contents = [
        "py-modindex.html",
        "roadmap.html",
        "search.html",
        "api_reference.html",
        "_sources",
        "getting_started.html",
        ".doctrees",
        "example_gallery",
        "index.html",
        "_images",
        "contributing.html",
        "genindex.html",
        "release_notes.html",
        "searchindex.js",
        ".buildinfo",
        "objects.inv",
        "_static",
        "api",
        "_modules",
        "_downloads",
    ]
    build_contents = os.listdir(build_dir)
    for item in expected_build_contents:
        assert item in build_contents

    # Currently warnings are allowed. If we also want to error on warnings, enable the next lines
    # # Check if there are any warnings in the Sphinx build output
    # if "build succeeded" not in result.stdout.lower():
    #     pytest.fail("Documentation build completed, but there were errors or warnings.")
