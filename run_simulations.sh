#!/bin/bash

module -q load anaconda

Ns=(8 16 32 48 64) # Our sizes for lattice
Hs=(0 1 2)
stepss=(3000)
n_temps=(200)
t_start=0.3
t_end=10.5

for n_temp in ${n_temps[@]}
do
    for N in ${Ns[@]}
    do
        for H in ${Hs[@]}
        do
            for steps in ${stepss[@]}
            do
                python3 ./run_heisenberg.py --N $N --H $H --steps $steps --temp $t_start $t_end $n_temp 
            done
        done
    done
done

