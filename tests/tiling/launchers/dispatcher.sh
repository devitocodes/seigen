#!/bin/bash

if [ "$node" == "erebus" ]; then
    declare -a cache=(1.4 1.2)
    declare -a snmesh=(1.4 1.2)
    nodeid=0
elif [ "$node" == "cx1-ivy" ]; then
    declare -a cache=(1.0 0.8 0.6 0.4 0.2 0.1)
    declare -a snmesh=(1.0 0.8 0.6)
    nodeid=1
elif [ "$node" == "cx1-haswell" ]; then
    declare -a cache=(1.0 0.8 0.6 0.4)
    declare -a snmesh=(1.0 0.8 0.6)
    nodeid=2
elif [ "$node" == "cx2-westmere" ]; then
    if [ "$mode" == "multinode" ]; then
        echo "Illegal combination node=$node and mode=$mode"
        exit
    fi
    declare -a cache=(1.0 0.8 0.6 0.4 0.2 0.1)
    declare -a snmesh=(1.0)
    nodeid=3
elif [ "$node" == "cx2-sandyb" ]; then
    if [ "$mode" == "singlenode" ]; then
        echo "Illegal combination node=$node and mode=$mode"
        exit
    fi
    declare -a cache=(1.0 0.8 0.6 0.4)
    nodeid=4
elif [ "$node" == "cx2-haswell" ]; then
    if [ "$mode" == "singlenode" ]; then
        echo "Illegal combination node=$node and mode=$mode"
        exit
    fi
    declare -a cache=(1.0 0.8 0.6 0.4)
    nodeid=5
elif [ "$node" == "cx2-broadwell" ]; then
    if [ "$mode" == "singlenode" ]; then
        echo "Illegal combination node=$node and mode=$mode"
        exit
    fi
    declare -a cache=(1.0 0.8 0.6 0.4)
    nodeid=6
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
            elif [ "$node" == "cx2-westmere" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=72:00:00 -l select=1:ncpus=12:mpiprocs=12:mem=32gb launchers/cache_populator.sh
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
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=24:00:00 -l select=1:ncpus=20:mem=32gb:ivyb=true launchers/executor.sh
            elif [ "$node" == "cx1-haswell" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=24:00:00 -l select=1:ncpus=20:mem=32gb:icib=true -q pqcdt launchers/executor.sh
            elif [ "$node" == "cx2-westmere" ]; then
                qsub -v nodename=$nodeid,poly=$poly,h=$h -l walltime=24:00:00 -l select=1:ncpus=12:mpiprocs=12:mem=32gb launchers/executor.sh
            fi
        done
    done
elif [ "$mode" == "multinode" ]; then
    echo "Executing the test suite on a cluster of $node nodes"
    for poly in 1 2 3 4; do
        if [ "$node" == "cx1-ivy" ]; then
            qsub -v poly=$poly,h=1.0,nodename=$nodeid -l walltime=72:00:00 -l select=1:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.8,nodename=$nodeid -l walltime=72:00:00 -l select=2:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.6,nodename=$nodeid -l walltime=72:00:00 -l select=4:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
            qsub -v poly=$poly,h=0.4,nodename=$nodeid -l walltime=72:00:00 -l select=8:ncpus=20:mem=48gb:ivyb=true launchers/executor.sh
        elif [ "$node" == "cx1-haswell" ]; then
            qsub -v poly=$poly,h=1.0,nodename=$nodeid -l walltime=72:00:00 -l select=1:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.8,nodename=$nodeid -l walltime=72:00:00 -l select=2:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.6,nodename=$nodeid -l walltime=72:00:00 -l select=4:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
            qsub -v poly=$poly,h=0.4,nodename=$nodeid -l walltime=72:00:00 -l select=8:ncpus=20:mem=48gb:icib=true -q pqcdt launchers/executor.sh
        elif [ "$node" == "cx2-sandyb" ]; then
            echo "node=$node and mode=$mode not available yet"
            exit
        elif [ "$node" == "cx2-haswell" ]; then
            qsub -v poly=$poly,h=1.0,nodename=$nodeid,np=24 -l walltime=12:00:00 -l select=8:ncpus=24:mem=48gb -q hastest launchers/executor.sh
            qsub -v poly=$poly,h=0.8,nodename=$nodeid,np=48 -l walltime=24:00:00 -l select=8:ncpus=24:mem=48gb -q hastest launchers/executor.sh
            qsub -v poly=$poly,h=0.6,nodename=$nodeid,np=96 -l walltime=36:00:00 -l select=8:ncpus=24:mem=48gb -q hastest launchers/executor.sh
            qsub -v poly=$poly,h=0.4,nodename=$nodeid,np=192 -l walltime=48:00:00 -l select=8:ncpus=24:mem=48gb -q hastest launchers/executor.sh
        elif [ "$node" == "cx2-broadwell" ]; then
            echo "node=$node and mode=$mode not available yet"
            exit
        else
            echo "Cannot run multi-node experiments on $node"
            exit
        fi
    done
else
    echo "Unrecognized mode $mode"
    exit
fi
