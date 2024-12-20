import numpy as np
from multiprocessing import Pool

def compute_next_state(args):
    grid, x, y = args
    rows, cols = grid.shape
    neighbors = [
        grid[(x-1) % rows, (y-1) % cols], grid[(x-1) % rows, y], grid[(x-1) % rows, (y+1) % cols],
        grid[x, (y-1) % cols],                             grid[x, (y+1) % cols],
        grid[(x+1) % rows, (y-1) % cols], grid[(x+1) % rows, y], grid[(x+1) % rows, (y+1) % cols]
    ]
    alive_neighbors = sum(neighbors)

    # Apply Conway's Game of Life rules
    if grid[x, y] == 1:  # Alive cell
        return 1 if alive_neighbors in (2, 3) else 0
    else:  # Dead cell
        return 1 if alive_neighbors == 3 else 0

def update_grid(grid):
    rows, cols = grid.shape
    with Pool() as pool:
        args = [(grid, x, y) for x in range(rows) for y in range(cols)]
        results = pool.map(compute_next_state, args)
    return np.array(results).reshape(grid.shape)

def main():
    rows, cols = 10, 10
    grid = np.random.choice([0, 1], size=(rows, cols))

    print(\"Initial Grid:\")
    print(grid)

    for step in range(5):
        grid = update_grid(grid)
        print(f\"Step {step + 1}:\")
        print(grid)

if __name__ == \"__main__\":
    main()
