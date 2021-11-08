# +
import multiprocessing
import sys
import os
from time import time
from typing import Tuple
from socket import gethostname

import numpy as np
import matplotlib.pyplot as plt

from numba import njit

# import correct tqdm version if inside Jupyter notebook
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm
# -

np.random.seed(444222)


## initialize a random grid of spins of -1 and 1
def initialize(N):
    return np.random.uniform(low=-1.0, high=1.0, size=(N, N, 3))


@njit(fastmath=True)
def symmetric_spin(s):
    """
    Ensure that the change in a spin direction is symmetrically
    distributed around the current spin direction.

    Args:
        s (1x3 ndarray): spin s_i

    Returns:
        ds: the nudge to spin s_i
    """
    # Numba does not support the size-argument
    ds = np.array([np.random.uniform(-1.0, 1.0) for _ in range(3)])
    ds_max = np.max(np.abs(s - ds))

    while np.linalg.norm(ds) > ds_max:
        ds = np.array([np.random.uniform(-1.0, 1.0) for _ in range(3)])
        ds_max = np.max(np.abs(s - ds))

    return ds


## metropolis-hastings step to determine a random spin to flip and see if the flip is valid
@njit(fastmath=True)
def metropolis(grid, beta, H):
    N = grid.shape[0]
    for _ in range(N * N):
        pos = np.random.randint(0, N, size=2)
        nbrs = grid[(pos[0] + 1) % N, pos[1]] + grid[(pos[0] - 1) % N, pos[1]] + \
               grid[pos[0], (pos[1] + 1) % N] + grid[pos[0], (pos[1] - 1) % N]

        dE = np.sum(2 * grid[pos[0], pos[1]] * (nbrs + H))

        ## flip if dE<0 or with prob exp^(-dE*beta)
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            s = grid[pos[0], pos[1]]
            proposal = s + symmetric_spin(s)
            proposal = proposal / np.linalg.norm(proposal)
            grid[pos[0], pos[1]] = proposal

    return grid


## compute the energy of the current spin configuration
@njit(fastmath=True)
def energy(grid, H):
    E = 0
    N = grid.shape[0]
    for x in range(N):
        for y in range(N):
            nbrs = grid[(x + 1) % N, y] + grid[(x - 1) % N, y] + grid[x, (y + 1) % N] + grid[x, (y - 1) % N]
            E -= np.sum((nbrs + H) * grid[x, y])
    return 1.0 * E / 4  ## avoid overcounting


@njit(fastmath=True)
def magnetization(grid):
    return np.sum(grid)


# +
def plot_system(grid, t, T, H):
    fig = plt.figure(t + 1, figsize=(12, 8))
    plt.imshow(np.copy(grid), interpolation='nearest', cmap='binary', vmin=-1, vmax=1, origin='lower')
    plt.title(f"System at time={t}, T={T:.2f}, external field H={H:.2f}")
    plt.grid()

    ## uncomment if you want to save the system configurations
    # np.savetxt("configuration_ising_2d_{}_{}_{}_{}.dat".format(grid.shape[0],t,T,H), grid)

    return fig


def _start(settings: Tuple):
        """ Run the routine for temperature input (temp, N, H, steps) and return E, M, C, X"""
        T, N, H, steps = settings
        snaps = []
            
        # for ii, T in enumerate(temp):
        E1=0
        M1=0
        E2=0
        M2=0
        grid = initialize(N) ## get the initial configuration
        beta = 1.0/T ## k_B = 1  

        ## parameters to calculate running average (notice that these are averages per spin)
        n1 = 1.0/(steps*N*N)
        n2 = 1.0/(steps*steps*N*N)

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
    

def run_simulation(N: float, H: float, steps: float, temp: np.ndarray):
    """
    Run the Metropolis-Hastings algorithm for the 2D lattice of 3D spins with the given settings.
    
    `n_temp` is inferred from len(temp)
    
    Arguments:
        N: size of lattice
        H: external magnetic field
        steps: number of steps to take for equilibrium
        temp: list of temperatures to run simulation for
    
    Returns:
        (E, M, C, X, wall_time):
            Energy, magnetisation, specific heat, susceptibility as vectors for temp.
            wall_time given as scalar of seconds taken for computation.
        
    """
    data_path = './results'
    n_temp = len(temp)

    ## small sanity check on input parameters
    if N<2 or steps<1 or temp[0]<0:
        raise ValueError("Invalid command line parameters")


    # Run the routine in parallel (NB: might not work on windows)
    t0 = time()
    
    settings = [(T, N, H, steps) for T in temp]

    # Run without parallel processing if on Windows
    if os.name == 'nt':
        results = list(tqdm(map(_start, settings)))
        E, M, C, X = np.array(results).T
    else:
        print(f'Using {os.cpu_count()} threads on {gethostname()}')
        with multiprocessing.Pool() as pool:
            results = list(tqdm(pool.imap(_start, settings), total=n_temp))
            E, M, C, X = np.array(results).T

    wall_time = time() - t0
    print(f'Took {wall_time} s')
    
    
    return E, M, C, X, wall_time
