#!/bin/bash

module -q load anaconda

Ns=(8 16 32 48 64) # Our sizes for lattice

Hxs=(0 1 1 0)
Hys=(0.5 0 1 0)
Hzs=(0.86 0 1 0)

stepss=(4000)

n_temps=(400)
t_start=0.2
t_end=10

for n_temp in ${n_temps[@]}
do
    for N in ${Ns[@]}
    do
        for ((i=0; i<${#Hxs[*]}; ++i))
        do
            for steps in ${stepss[@]}
            do
                python3 ./run_heisenberg.py --N $N --H ${Hxs[$i]} ${Hys[$i]} ${Hzs[$i]} --steps $steps --temp $t_start $t_end $n_temp 
            done
        done
    done
done

