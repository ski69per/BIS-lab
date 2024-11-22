import numpy as np

# Objective function (Sphere function)
def objective_function(x):
    return np.sum(x**2)

# Parameters
N, D, max_iter = 20, 2, 100
bounds = [-10, 10]

# Initialize wolves
positions = np.random.uniform(bounds[0], bounds[1], (N, D))
alpha, beta, delta = np.inf, np.inf, np.inf
alpha_pos, beta_pos, delta_pos = None, None, None

# GWO loop
for t in range(max_iter):
    # Update fitness and leaders
    for i in range(N):
        fitness = objective_function(positions[i])
        if fitness < alpha: 
            alpha, beta, delta = fitness, alpha, beta
            alpha_pos, beta_pos, delta_pos = positions[i], alpha_pos, beta_pos
        elif fitness < beta:
            beta, delta = fitness, beta
            beta_pos, delta_pos = positions[i], beta_pos
        elif fitness < delta:
            delta = fitness
            delta_pos = positions[i]

    # Update positions
    a = 2 - 2 * t / max_iter
    for i in range(N):
        A1, C1 = 2 * a * np.random.rand(D) - a, 2 * np.random.rand(D)
        A2, C2 = 2 * a * np.random.rand(D) - a, 2 * np.random.rand(D)
        A3, C3 = 2 * a * np.random.rand(D) - a, 2 * np.random.rand(D)
        
        X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - positions[i])
        X2 = beta_pos - A2 * np.abs(C2 * beta_pos - positions[i])
        X3 = delta_pos - A3 * np.abs(C3 * delta_pos - positions[i])
        
        positions[i] = np.clip((X1 + X2 + X3) / 3, bounds[0], bounds[1])

    print(f"Iteration {t+1}/{max_iter}, Best Score: {alpha}")

# Output best solution
print(f"\nOptimal Solution: Position {alpha_pos}, Value {alpha}")
