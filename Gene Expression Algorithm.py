import numpy as np
import random

# Objective function
def fitness_function(x, y):
    return -x**2 - y**2 + 4*x + 4*y

# Initialize population
def initialize_population(size, bounds):
    return [np.random.uniform(low=bounds[0], high=bounds[1], size=2) for _ in range(size)]

# Mutation
def mutate(individual, bounds, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.uniform(-0.5, 0.5)
            # Clamp to bounds
            individual[i] = np.clip(individual[i], bounds[0], bounds[1])
    return individual

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# Selection
def select_population(population, fitnesses, num_to_select):
    selected = np.random.choice(
        population, size=num_to_select, replace=False, p=fitnesses / sum(fitnesses)
    )
    return list(selected)

# Gene Expression Algorithm
def gene_expression_algorithm(bounds, population_size, generations):
    # Initialize population
    population = initialize_population(population_size, bounds)
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = np.array([fitness_function(ind[0], ind[1]) for ind in population])
        
        # Selection
        selected_population = select_population(population, fitnesses, population_size // 2)
        
        # Crossover and Mutation
        next_generation = []
        while len(next_generation) < population_size:
            parents = random.sample(selected_population, 2)
            child1, child2 = crossover(parents[0], parents[1])
            next_generation.append(mutate(child1, bounds))
            next_generation.append(mutate(child2, bounds))
        
        population = next_generation[:population_size]
        
        # Logging best solution
        best_fitness = max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
    
    return best_individual, best_fitness

# Parameters
bounds = [-10, 10]  # Bounds for x and y
population_size = 20
generations = 50

# Run GEA
best_solution, best_fitness = gene_expression_algorithm(bounds, population_size, generations)
print(f"\nBest Solution: {best_solution}, Fitness: {best_fitness}")
