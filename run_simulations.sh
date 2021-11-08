#!/bin/bash

module -q load anaconda

Ns=(10 25 50) # Our sizes for lattice
Hs=(1)
stepss=(2000)
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

