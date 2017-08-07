Run dispatcher.sh from the directory /path/to/seigen/tests/tiling as:

    node=X mode=Y ./launchers/dispatcher.sh

Where X can be:

    * erebus
    * cx1-ivy
    * cx1-haswell
    * cx2-westmere
    * cx2-sandyb
    * cx2-haswell
    * cx2-broadwell

And Y can be:

    * singlenode : to run the test suite on the single X node
    * multinode : to run the test suite on a cluster of X nodes

Some combinations are not valid:

    * cx2-westmere only works in singlenode mode
    * cx2-{sandyb,haswell,broadwell} only work in multinode mode

Examples:

    node=erebus mode=singlenode ./launchers/dispatcher.sh
    node=cx1-haswell mode=multinode ./launchers/dispatcher.sh
