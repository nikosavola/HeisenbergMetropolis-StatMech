#!/bin/bash

module -q load anaconda

n_temps=(200)
Ns=(10 25 50) # Our sizes for lattice
Hs=(1)
stepss=(2000)

for n_temp in ${n_temps[@]}
do
    for N in ${Ns[@]}
    do
        for H in ${Hs[@]}
        do
            for steps in ${stepss[@]}
            do
                python3 ./run_heisenberg.py --n_temp $n_temp --N $N --H $H --steps $steps
            done
        done
    done
done

