Run dispatcher.sh from the directory /path/to/seigen/tests/tiling as:

    node=X mode=Y ./launchers/dispatcher.sh

Where X can be:

    * erebus
    * cx1-ivy
    * cx1-haswell

And Y can be:

    * populator : to populate the Firedrake cache
    * singlenode : to run the test suite on the single node Y
    * multinode : to run the test suite on a cluster of Y nodes

Examples:

    node=erebus mode=singlenode ./launchers/dispatcher.sh
    node=cx1-haswell mode=multinode ./launchers/dispatcher.sh
