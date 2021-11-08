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

from heisenberg_2d import initialize, metropolis, energy, magnetization, run_simulation


# read arguments
import argparse

if __name__ == '__main__':
    """
    Run as::
        
        python3 run_heisenberg.py --N 10 --H 1 --steps 4000 --temp 0.3 10.5 500
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', nargs="+",  metavar='float', type=float)
    parser.add_argument('--N', metavar='int', type=int)
    parser.add_argument('--H', metavar='int', type=int)
    parser.add_argument('--steps', metavar='int', type=int)

    args = parser.parse_args()
    N, H, steps, temp = args.N, args.H, args.steps, args.temp
    print(args)

    temp = np.linspace(temp[0], temp[1], int(temp[2]))
    n_temp = len(temp)
    
    snaps = []
    data_path = './results'
    
    
    ## Run the simulations
    *results, wall_time = run_simulation(N, H, steps, temp)
    results = np.array(results).T
    
    
    ## Save the results
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


