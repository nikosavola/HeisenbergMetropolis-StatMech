import multiprocessing
import sys
import os
from pathlib import Path
from time import time
from socket import gethostname

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from tqdm import tqdm

from heisenberg_2d import initialize, metropolis, energy, magnetization


# read arguments
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_temp', metavar='int', type=int)
    parser.add_argument('--N', metavar='int', type=int)
    parser.add_argument('--H', metavar='int', type=int)
    parser.add_argument('--steps', metavar='int', type=int)

    args = parser.parse_args()
    n_temp, N, H, steps = args.n_temp, args.N, args.H, args.steps
    print(args)


    temp = np.linspace(0.3, 10.5, n_temp)
    snaps = []
    data_path = './results'


    ## small sanity check on input parameters
    if N<2 or steps<1 or temp[0]<0:
        print("Invalid command line parameters")

    ## parameters to calculate running average (notice that these are averages per spin)
    n1 = 1.0/(steps*N*N)
    n2 = 1.0/(steps*steps*N*N)

    def start(T: float):
        """ Run the routine for temperature T and return E, M, C, X"""
    # for ii, T in enumerate(temp):
        E1=0
        M1=0
        E2=0
        M2=0
        grid = initialize(N) ## get the initial configuration
        beta = 1.0/T ## k_B = 1  


        ## first we equilibrate the system 
        ## (assumption is that snapshots are wanted here)
        for t in range(steps):
            if t in snaps:
                plot_system(grid, t, T, H)

            metropolis(grid, beta, H)

        ## then we start to actually collect data, if we aren't just plotting snapshots
        if len(snaps)==0:
            for t in range(steps):
                metropolis(grid, beta, H)
                tE = energy(grid, H)
                tM = magnetization(grid)

                E1 += tE
                E2 += tE*tE
                M1 += tM
                M2 += tM*tM

            E = n1*E1
            M = n1*M1
            C = beta*beta*(n1*E2 - n2*E1*E1)
            X = beta*(n1*M2 - n2*M1*M1)

        return E, M, C, X



    # Run the routine in parallel (NB: might not work on windows)
    t0 = time()

    # Run without parallel processing if on Windows
    if os.name == 'nt':
        results = list(tqdm(map(start, temp)))
        E, M, C, X = np.array(results).T
    else:
        print(f'Using {os.cpu_count()} threads on {gethostname()}')
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(start, temp), total=n_temp))
            E, M, C, X = np.array(results).T

    wall_time = time() - t0
    print(f'Took {wall_time} s')



    # mkdir in case
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # gather results in a table just in case
    df = pd.DataFrame(results, columns=['E', 'M', 'C', 'X'])
    df['temp'] = temp
    for name, var in zip(['n_temp', 'N', 'steps', 'H', 'wall_time'], [n_temp, N, steps, H, wall_time]):
        df[name] = var

    # save the table
    df.to_csv(f'{data_path}/data_{n_temp}_{N}_{steps}_{H}.csv', sep=',', header=True, index=False)