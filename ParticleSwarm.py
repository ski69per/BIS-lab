import random
import numpy as np

def objective_function(x):
    # Example: Sphere function
    return sum(x_i ** 2 for x_i in x)

class Particle:
    def __init__(self, dimensions, bounds):
        self.position = np.array([random.uniform(bounds[dim][0], bounds[dim][1]) for dim in range(dimensions)])
        self.velocity = np.array([random.uniform(-1, 1) for _ in range(dimensions)])
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')
        self.current_value = float('inf')

    def update_personal_best(self):
        if self.current_value < self.best_value:
            self.best_value = self.current_value
            self.best_position = np.copy(self.position)

class PSO:
    def __init__(self, objective_function, dimensions, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter

        # Initialize particles
        self.particles = [Particle(dimensions, bounds) for _ in range(num_particles)]
        self.global_best_position = np.zeros(dimensions)
        self.global_best_value = float('inf')

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                # Evaluate the objective function
                particle.current_value = self.objective_function(particle.position)
                particle.update_personal_best()

                # Update global best if necessary
                if particle.current_value < self.global_best_value:
                    self.global_best_value = particle.current_value
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                # Update velocity and position
                r1, r2 = random.random(), random.random()
                cognitive_component = self.c1 * r1 * (particle.best_position - particle.position)
                social_component = self.c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = self.w * particle.velocity + cognitive_component + social_component

                # Update position and apply bounds
                particle.position += particle.velocity
                for dim in range(self.dimensions):
                    particle.position[dim] = np.clip(particle.position[dim], self.bounds[dim][0], self.bounds[dim][1])

            print(f"Iteration {iteration + 1}/{self.max_iter}, Best Value: {self.global_best_value}")

        return self.global_best_position, self.global_best_value

# Example usage
dimensions = 2
bounds = [(-10, 10), (-10, 10)]
pso = PSO(objective_function, dimensions, bounds, num_particles=30, max_iter=100)
best_position, best_value = pso.optimize()

print(f"Optimal Position: {best_position}")
print(f"Optimal Value: {best_value}")
