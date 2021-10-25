import matplotlib.pyplot as plt
import numpy as np

np.random.seed(444222)


## initialize a random grid of spins of -1 and 1
def initialize(N):
    return 2 * np.random.randint(2, size=(N, N)) - 1


## metropolis-hastings step to determine a random spin to flip and see if the flip is valid
def metropolis(grid, beta, H):
    N = grid.shape[0]
    for _ in range(N * N):
        pos = np.random.randint(0, N, size=2)
        nbrs = grid[(pos[0] + 1) % N, pos[1]] + grid[(pos[0] - 1) % N, pos[1]] + grid[pos[0], (pos[1] + 1) % N] + grid[
            pos[0], (pos[1] - 1) % N]

        dE = 2 * grid[pos[0], pos[1]] * (nbrs + H)

        ## flip if dE<0 or with prob exp^(-dE*beta)
        if dE < 0 or np.random.rand() < np.exp(-dE * beta):
            grid[pos[0], pos[1]] *= -1

    return grid


## compute the energy of the current spin configuration
def energy(grid, H):
    E = 0
    N = grid.shape[0]
    for x in range(N):
        for y in range(N):
            nbrs = grid[(x + 1) % N, y] + grid[(x - 1) % N, y] + grid[x, (y + 1) % N] + grid[x, (y - 1) % N]
            E -= (nbrs + H) * grid[x, y]
    return 1.0 * E / 4  ## avoid overcounting


def magnetization(grid):
    return np.sum(grid)


def plot_system(grid, t, T, H):
    fig = plt.figure(t + 1, figsize=(12, 8))
    plt.imshow(np.copy(grid), interpolation='nearest', cmap='binary', vmin=-1, vmax=1, origin='lower')
    plt.title("System at time={}, T={:.2f}, external field H={:.2f}".format(t, T, H))
    plt.grid()

    ## uncomment if you want to save the system configurations
    # np.savetxt("configuration_ising_2d_{}_{}_{}_{}.dat".format(grid.shape[0],t,T,H), grid)

    return fig
