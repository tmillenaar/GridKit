#!/bin/bash

# Allow script to finish even if one docs build has an error
set +e

# Save the current branch to return to later
current_branch=$(git rev-parse --abbrev-ref HEAD)
rootdir="$PWD"

# Get all tags starting with 'v', remove all tags before v0.7.0 and sort with largest version at the top.
# Any version before v0.7.0 does not properly build the documentation
tags=$(git tag | grep '^v' | sort -V | awk 'BEGIN{keep=0} $0=="v0.7.0"{keep=1} keep' | sort -Vr)

# Add dev tag which will represent the head of the main branch
tags="dev
$tags"

echo "Building docs for the following versions: $tags"

# Export tags to be used in conf.py
export GRIDKIT_DOC_BUILD_ROOTDIR=$rootdir
export GRIDKIT_DOC_BUILD_TAGS=$tags

mkdir -p /tmp/gridkit_docs
for tag in $tags; do
    echo "Start building docs for version: $tag"

    html_destination="$rootdir/build/sphinx/html/versions/$tag"
    echo "Building docs to $html_destination"

    # Check out code of tagged commit in /tmp folder
    git_dir=/tmp/gridkit_docs/$tag
    echo "Building $tag in $git_dir"

    rm -rf $git_dir
    git clone "${rootdir}/.." $git_dir

    if [ "$tag" = "dev" ]; then
        git -C $git_dir fetch origin main
        git -C $git_dir checkout origin/main
    else
        git -C $git_dir checkout $tag
    fi

    # Older structure did not have gridkit-py, add it to workdir if it exists
    if [ -d "$git_dir/gridkit-py" ]; then
      tag_workdir="$git_dir/gridkit-py"
    else
      tag_workdir="$git_dir"
    fi

    # Copy critical sphinx files that might not be up to date in older versions
    cp "${rootdir}/docs/source/conf.py" "${tag_workdir}/docs/source/conf.py"
    mkdir -p "${tag_workdir}/docs/source/_templates"
    cp "${rootdir}/docs/source/_templates/versions.html" "${tag_workdir}/docs/source/_templates/versions.html"
    cp "${rootdir}/docs/source/_templates/layout.html" "${tag_workdir}/docs/source/_templates/layout.html"

    # Make sure the docs report the tagged version
    sed -i -r "s|\"[[:alnum:]]+\.[[:alnum:]]+\.[[:alnum:]]+.+|\"$tag\"|" $tag_workdir/gridkit/version.py

    # Create venv in tmp folder
    python3 -m venv "$tag_workdir/venv"
    source "${tag_workdir}/venv/bin/activate"

    # Use pypi-timemachine to install the package dependencies in the state at the time of release
    if [ "$tag" = "dev" ]; then
        tag_date=$(date +%Y-%m-%d)
    else
        tag_date=$(git log -1 --format=%aI "$tag" | cut -d'T' -f1)
    fi
    yy_doy=$(date -d "$tag_date" +%y%j)  # use shorter representation as port, use year and day of year (out of 365)
    echo "Limiting PyPI to date: $tag_date"
    pip install pypi-timemachine
    pypi-timemachine --port $yy_doy $tag_date &  # Use trailing & to save pid
    PTM_PID=$!  # save pid to kill background process later

    # Install package and dependencies using the time boxed pypi proxy
    pip install -e $tag_workdir/[doc] --index-url "http://127.0.0.1:${yy_doy}/"

    # Build the docs for this version
    export GRIDKIT_DOC_BUILD_CURRENT_VERSION=$tag
    sphinx-build ${tag_workdir}/docs/source $html_destination

    # Clean up this version's build reamnants
    kill "$PTM_PID"  # Quit the localhost supporting the pypi-timemachine
    rm -rf $tag_workdir
done

# Link landing page to most recent version
ln -sf "${rootdir}/build/sphinx/html/versions/${GRIDKIT_DOC_BUILD_LATEST_VERSION}/index.html" "${rootdir}/build/sphinx/html/index.html"

# Left over cleanup
rm -rf /tmp/gridkit_docs
