#!/bin/bash

if [ "$node" == "erebus" ]; then
    declare -a cache=(1.2 1.4)
    declare -a snmesh=(1.2 1.4)
    nodeid=0
elif [ "$node" == "cx1-ivy" ]; then
    declare -a cache=(0.6 0.45 0.3 0.225 0.15 0.115)
    declare -a snmesh=(0.6 0.8)
    nodeid=1
elif [ "$node" == "cx1-haswell" ]; then
    declare -a cache=(0.6 0.45 0.3 0.225 0.15 0.115)
    declare -a snmesh=(0.6 0.8)
    nodeid=2
else
    echo "Unrecognized node: $node"
    exit
fi

if [ "$mode" == "populator" ]; then
    echo "Populating cache for node $node"
    for poly in 1 2 3 4; do
        for h in ${cache[@]}; do
            if [ "$node" == "erebus" ]; then
                nodename=$nodeid poly=$poly h=$h ./launchers/cache_populator.sh
            elif [ "$node" == "cx1-ivy" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=72:00:00 -l select=1:ncpus=20:mem=60gb:ivyb=true launchers/cache_populator.sh
            elif [ "$node" == "cx1-haswell" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=72:00:00 -l select=1:ncpus=20:mem=60gb -q pqcdt launchers/cache_populator.sh
            fi
        done
    done
elif [ "$mode" == 'singlenode' ]; then
    echo "Executing the test suite on node $node"
    for poly in 1 2 3 4; do
        for h in ${snmesh[@]}; do
            echo "Scheduling single node experiments: <poly=$poly,h=$h,node=$nome>"
            if [ "$node" == "erebus" ]; then
                nodename=$nodeid poly=$poly h=$h ./launchers/executor.sh
            elif [ "$node" == "cx1-ivy" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true launchers/executor.sh
            elif [ "$node" == "cx1-haswell" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=72:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt launchers/executor.sh
            fi
        done
    done
elif [ "$mode" == "multinode" ]; then
    echo "Executing the test suite on a cluster of $node nodes"
    for poly in 1 2 3 4; do
        if [ "$node" == "cx1-ivy" ]; then
            qsub -v poly=$poly,h=0.6,nodename=$nodeid -l walltime=72:00:00 -l select=1:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.45,nodename=$nodeid -l walltime=72:00:00 -l select=2:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.3,nodename=$nodeid -l walltime=72:00:00 -l select=4:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.225,nodename=$nodeid -l walltime=72:00:00 -l select=8:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.15,nodename=$nodeid -l walltime=72:00:00 -l select=16:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.115,nodename=$nodeid -l walltime=72:00:00 -l select=32:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
        elif [ "$node" == "cx1-haswell" ]; then
            qsub -v poly=$poly,h=0.6,nodename=$nodeid -l walltime=72:00:00 -l select=1:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.45,nodename=$nodeid -l walltime=72:00:00 -l select=2:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.3,nodename=$nodeid -l walltime=72:00:00 -l select=4:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.225,nodename=$nodeid -l walltime=72:00:00 -l select=8:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
        else
            echo "Cannot run multi-node experiments on $node"
            exit
        fi
    done
else
    echo "Unrecognized mode $mode"
    exit
fi
