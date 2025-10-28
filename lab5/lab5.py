import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd

# Goldstein-Price function
def goldstein_price(x):
    x1, x2 = x
    term1 = (1 + (x1 + x2 + 1)**2 * 
            (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * 
            (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

# ============================================================================
# FIXED IMPLEMENTATION OF DIFFERENT ES STRATEGIES
# ============================================================================

class Individual:
    """Individual with strategic parameters as per equation (5.1)"""
    def __init__(self, x, sigma):
        self.x = np.array(x, dtype=np.float64)
        self.sigma = np.array(sigma, dtype=np.float64)
        self.fitness = None
    
    def __str__(self):
        return f"x: {self.x}, σ: {self.sigma}, fitness: {self.fitness}"

def one_plus_one_es(objective_func, bounds, max_generations=500, initial_sigma=0.3, 
                   success_rule_interval=50, verbose=True):
    """(1+1)-ES: Twofold evolutionary strategy"""
    dim = len(bounds)
    fitness_history = []
    sigma_history = []
    all_populations = []
    
    # Initialize single parent
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    sigma = np.full(dim, initial_sigma)
    parent = Individual(x, sigma)
    parent.fitness = objective_func(parent.x)
    
    success_count = 0
    all_populations.append([parent.x.copy()])
    
    if verbose:
        print(f"(1+1)-ES: Initial fitness = {parent.fitness:.6f}")
    
    for generation in range(max_generations):
        # Create offspring by mutation
        offspring_x = parent.x + parent.sigma * np.random.normal(0, 1, dim)
        
        # Apply bounds
        for i in range(dim):
            offspring_x[i] = np.clip(offspring_x[i], bounds[i][0], bounds[i][1])
        
        offspring_fitness = objective_func(offspring_x)
        
        # Selection: replace if better
        if offspring_fitness < parent.fitness:
            parent.x = offspring_x.copy()
            parent.fitness = offspring_fitness
            success_count += 1
        
        fitness_history.append(parent.fitness)
        sigma_history.append(parent.sigma.copy())
        all_populations.append([parent.x.copy()])
        
        # 1/5 Success Rule
        if generation > 0 and generation % success_rule_interval == 0:
            success_rate = success_count / success_rule_interval
            if success_rate > 0.2:
                parent.sigma *= 1.22
            elif success_rate < 0.2:
                parent.sigma *= 0.82
            success_count = 0
            
    return parent.x, parent.fitness, fitness_history, sigma_history, all_populations

def mu_comma_lambda_es(objective_func, bounds, mu=15, lambda_=45, max_generations=100, 
                      initial_sigma=0.3, verbose=True):
    """(μ,λ)-ES: Multiple evolutionary strategy"""
    dim = len(bounds)
    population = []
    fitness_history = []
    all_populations = []
    
    # Initialize population
    for _ in range(mu):
        x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        sigma = np.full(dim, initial_sigma)
        ind = Individual(x, sigma)
        ind.fitness = objective_func(x)
        population.append(ind)
    
    population.sort(key=lambda ind: ind.fitness)
    best_individual = population[0]
    fitness_history.append(best_individual.fitness)
    all_populations.append([ind.x.copy() for ind in population])
    
    if verbose:
        print(f"(μ,λ)-ES: Initial best fitness = {best_individual.fitness:.6f}")
    
    for generation in range(max_generations):
        offspring = []
        
        # Generate λ offspring
        for _ in range(lambda_):
            # Select parents using tournament selection
            parents = []
            for _ in range(2):
                candidates = np.random.choice(population, min(3, len(population)), replace=False)
                best_candidate = min(candidates, key=lambda ind: ind.fitness)
                parents.append(best_candidate)
            
            # Recombination
            x_child = np.mean([p.x for p in parents], axis=0)
            sigma_child = np.mean([p.sigma for p in parents], axis=0)
            
            # Mutation
            tau = 1.0 / np.sqrt(2 * np.sqrt(dim))
            sigma_child = sigma_child * np.exp(tau * np.random.normal(0, 1, dim))
            sigma_child = np.clip(sigma_child, 0.01, 2.0)
            
            x_child = x_child + sigma_child * np.random.normal(0, 1, dim)
            
            # Apply bounds
            for i in range(dim):
                x_child[i] = np.clip(x_child[i], bounds[i][0], bounds[i][1])
            
            child = Individual(x_child, sigma_child)
            child.fitness = objective_func(child.x)
            offspring.append(child)
        
        # Selection
        offspring.sort(key=lambda ind: ind.fitness)
        population = offspring[:mu]
        
        best_individual = population[0]
        fitness_history.append(best_individual.fitness)
        all_populations.append([ind.x.copy() for ind in population])
        
    return best_individual.x, best_individual.fitness, fitness_history, all_populations

def mu_plus_lambda_es(objective_func, bounds, mu=15, lambda_=45, max_generations=100,
                     initial_sigma=0.3, verbose=True):
    """(μ+λ)-ES: Multiple evolutionary strategy"""
    dim = len(bounds)
    population = []
    fitness_history = []
    all_populations = []
    
    # Initialize population
    for _ in range(mu):
        x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        sigma = np.full(dim, initial_sigma)
        ind = Individual(x, sigma)
        ind.fitness = objective_func(x)
        population.append(ind)
    
    population.sort(key=lambda ind: ind.fitness)
    best_individual = population[0]
    fitness_history.append(best_individual.fitness)
    all_populations.append([ind.x.copy() for ind in population])
    
    if verbose:
        print(f"(μ+λ)-ES: Initial best fitness = {best_individual.fitness:.6f}")
    
    for generation in range(max_generations):
        offspring = []
        
        # Generate λ offspring
        for _ in range(lambda_):
            # Select parents
            parents = []
            for _ in range(2):
                candidates = np.random.choice(population, min(3, len(population)), replace=False)
                best_candidate = min(candidates, key=lambda ind: ind.fitness)
                parents.append(best_candidate)
            
            # Recombination
            x_child = np.mean([p.x for p in parents], axis=0)
            sigma_child = np.mean([p.sigma for p in parents], axis=0)
            
            # Mutation
            tau = 1.0 / np.sqrt(2 * np.sqrt(dim))
            sigma_child = sigma_child * np.exp(tau * np.random.normal(0, 1, dim))
            sigma_child = np.clip(sigma_child, 0.01, 2.0)
            
            x_child = x_child + sigma_child * np.random.normal(0, 1, dim)
            
            # Apply bounds
            for i in range(dim):
                x_child[i] = np.clip(x_child[i], bounds[i][0], bounds[i][1])
            
            child = Individual(x_child, sigma_child)
            child.fitness = objective_func(child.x)
            offspring.append(child)
        
        # Selection from parents + offspring
        combined = population + offspring
        combined.sort(key=lambda ind: ind.fitness)
        population = combined[:mu]
        
        best_individual = population[0]
        fitness_history.append(best_individual.fitness)
        all_populations.append([ind.x.copy() for ind in population])
        
    return best_individual.x, best_individual.fitness, fitness_history, all_populations

# ============================================================================
# FIXED VISUALIZATION FUNCTIONS
# ============================================================================

def plot_function_with_trajectory(population_history, strategy_name, bounds):
    """Plot function with optimization trajectory - FIXED VERSION"""
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = goldstein_price([X1[j, i], X2[j, i]])
    
    fig = plt.figure(figsize=(18, 6))
    
    # 3D surface plot with trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
    
    # Plot optimization trajectory with FIXED color mapping
    for gen, population in enumerate(population_history):
        if gen % max(1, len(population_history)//10) == 0:  # Sample generations
            color_val = gen / len(population_history)
            for individual in population:
                fitness = goldstein_price(individual)
                ax1.scatter(individual[0], individual[1], fitness, 
                           color=plt.cm.plasma(color_val), s=30, alpha=0.7)
    
    # Mark important points
    if population_history:
        # Start points
        for individual in population_history[0]:
            fitness = goldstein_price(individual)
            ax1.scatter(individual[0], individual[1], fitness, 
                       color='green', s=100, marker='o', label='Start' if individual is population_history[0][0] else "")
        
        # End points  
        for individual in population_history[-1]:
            fitness = goldstein_price(individual)
            ax1.scatter(individual[0], individual[1], fitness, 
                       color='red', s=100, marker='*', label='End' if individual is population_history[-1][0] else "")
    
    # Mark global optimum
    ax1.scatter(0, -1, 3, color='gold', s=200, marker='D', label='Global Optimum', edgecolors='black')
    
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2)')
    ax1.set_title(f'{strategy_name}\n3D View with Optimization Trajectory')
    ax1.legend()
    
    # 2D contour plot with trajectory
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X1, X2, Z, levels=20, alpha=0.6)
    plt.colorbar(contour, ax=ax2)
    
    # Plot trajectory in 2D with FIXED color mapping
    for gen, population in enumerate(population_history):
        if gen % max(1, len(population_history)//10) == 0:
            color_val = gen / len(population_history)
            x1_vals = [ind[0] for ind in population]
            x2_vals = [ind[1] for ind in population]
            ax2.scatter(x1_vals, x2_vals, color=plt.cm.plasma(color_val), 
                       alpha=0.6, s=20)
    
    # Mark important points
    ax2.scatter(0, -1, color='gold', s=200, marker='D', label='Global Optimum', edgecolors='black')
    
    if population_history:
        start_x1 = [ind[0] for ind in population_history[0]]
        start_x2 = [ind[1] for ind in population_history[0]]
        end_x1 = [ind[0] for ind in population_history[-1]]
        end_x2 = [ind[1] for ind in population_history[-1]]
        
        ax2.scatter(start_x1, start_x2, color='green', s=80, marker='o', label='Initial Population')
        ax2.scatter(end_x1, end_x2, color='red', s=80, marker='*', label='Final Population')
    
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title(f'{strategy_name}\nContour with Search Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence plot
    ax3 = fig.add_subplot(133)
    best_fitness_history = [min([goldstein_price(ind) for ind in pop]) for pop in population_history]
    ax3.plot(best_fitness_history, 'b-', linewidth=2, label='Best Fitness')
    ax3.axhline(y=3, color='r', linestyle='--', label='Theoretical Minimum')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitness')
    ax3.set_title('Convergence History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if min(best_fitness_history) > 0:
        ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_initial_function():
    """Plot just the Goldstein-Price function without trajectories"""
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = goldstein_price([X1[j, i], X2[j, i]])
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(0, -1, 3, color='red', s=100, label='Global Optimum')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2)')
    ax1.set_title('Goldstein-Price Function (3D)')
    ax1.legend()
    
    # Contour plot
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X1, X2, Z, levels=20)
    ax2.scatter(0, -1, color='red', s=100, label='Global Optimum')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Contour Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2)
    
    # Function description
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.text(0.1, 0.9, 'Goldstein-Price Function', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.7, 'Global Minimum: f(0, -1) = 3.0', fontsize=12)
    ax3.text(0.1, 0.6, 'Search Space: -2 ≤ x₁, x₂ ≤ 2', fontsize=12)
    ax3.text(0.1, 0.5, 'Characteristics:', fontsize=12, fontweight='bold')
    ax3.text(0.1, 0.4, '- Multiple local minima', fontsize=10)
    ax3.text(0.1, 0.3, '- Complex landscape', fontsize=10)
    ax3.text(0.1, 0.2, '- Challenging for optimization', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def compare_es_strategies(objective_func, bounds):
    """Comprehensive comparison of different ES strategies"""
    print("=" * 70)
    print("COMPREHENSIVE ES STRATEGIES COMPARISON")
    print("=" * 70)
    
    strategies = {
        '(1+1)-ES': one_plus_one_es,
        '(μ,λ)-ES': mu_comma_lambda_es,
        '(μ+λ)-ES': mu_plus_lambda_es
    }
    
    results = {}
    
    for name, strategy_func in strategies.items():
        print(f"\n--- Running {name} ---")
        start_time = time.time()
        
        try:
            if name == '(1+1)-ES':
                best_x, best_fitness, fitness_history, sigma_history, pop_history = strategy_func(
                    objective_func, bounds, max_generations=200, verbose=False
                )
            else:
                best_x, best_fitness, fitness_history, pop_history = strategy_func(
                    objective_func, bounds, mu=15, lambda_=45, max_generations=100, verbose=False
                )
            
            end_time = time.time()
            
            results[name] = {
                'best_solution': best_x,
                'best_fitness': best_fitness,
                'fitness_history': fitness_history,
                'computation_time': end_time - start_time,
                'generations': len(fitness_history),
                'error': np.linalg.norm(best_x - np.array([0, -1])),
                'population_history': pop_history
            }
            
            print(f"Best fitness: {best_fitness:.8f}")
            print(f"Best solution: x1={best_x[0]:.6f}, x2={best_x[1]:.6f}")
            print(f"Computation time: {end_time - start_time:.4f}s")
            print(f"Generations: {len(fitness_history)}")
            print(f"Error from optimum: {results[name]['error']:.8f}")
            
            # Plot trajectory for this strategy
            print(f"Plotting trajectory for {name}...")
            plot_function_with_trajectory(pop_history, name, bounds)
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def parameter_sensitivity_analysis(objective_func, bounds):
    """Comprehensive parameter sensitivity analysis"""
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Test different population sizes
    print("\n1. Population Size Sensitivity (for (μ,λ)-ES):")
    pop_sizes = [5, 10, 20, 30]
    pop_results = {}
    
    for pop_size in pop_sizes:
        start_time = time.time()
        best_x, best_fitness, fitness_history, _ = mu_comma_lambda_es(
            objective_func, bounds, mu=pop_size, lambda_=pop_size*3, 
            max_generations=80, verbose=False
        )
        end_time = time.time()
        
        pop_results[pop_size] = {
            'fitness': best_fitness,
            'time': end_time - start_time,
            'generations': len(fitness_history),
            'solution': best_x
        }
        
        print(f"Population {pop_size}: Fitness = {best_fitness:.6f}, "
              f"Time = {end_time - start_time:.3f}s")
    
    # Test different mutation parameters
    print("\n2. Initial Mutation Strength Sensitivity (for (1+1)-ES):")
    sigma_values = [0.1, 0.3, 0.5, 0.8]
    sigma_results = {}
    
    for sigma in sigma_values:
        start_time = time.time()
        best_x, best_fitness, fitness_history, _, _ = one_plus_one_es(
            objective_func, bounds, initial_sigma=sigma, 
            max_generations=200, verbose=False
        )
        end_time = time.time()
        
        sigma_results[sigma] = {
            'fitness': best_fitness,
            'time': end_time - start_time,
            'generations': len(fitness_history),
            'solution': best_x
        }
        
        print(f"Sigma {sigma}: Fitness = {best_fitness:.6f}, "
              f"Time = {end_time - start_time:.3f}s")
    
    return pop_results, sigma_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("COMPREHENSIVE EVOLUTIONARY STRATEGY ANALYSIS")
    print("Goldstein-Price Function Optimization")
    print("=" * 70)
    
    # Known global minimum
    true_optimum = np.array([0, -1])
    true_minimum = 3.0
    
    print(f"Theoretical global minimum: f({true_optimum[0]}, {true_optimum[1]}) = {true_minimum}")
    
    # Define bounds
    bounds = [(-2, 2), (-2, 2)]
    
    # 1. Plot the initial function
    print("\n1. PLOTTING THE GOLDSTEIN-PRICE FUNCTION...")
    plot_initial_function()
    
    # 2. Compare all ES strategies with trajectory visualization
    print("\n2. COMPARING DIFFERENT ES STRATEGIES WITH TRAJECTORY VISUALIZATION...")
    results = compare_es_strategies(goldstein_price, bounds)
    
    # 3. Parameter sensitivity analysis
    print("\n3. PERFORMING PARAMETER SENSITIVITY ANALYSIS...")
    pop_results, sigma_results = parameter_sensitivity_analysis(goldstein_price, bounds)
    
    # 4. Display final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    if results:
        best_strategy = min(results.items(), key=lambda x: x[1]['best_fitness'])
        print(f"Best performing strategy: {best_strategy[0]}")
        print(f"Best fitness: {best_strategy[1]['best_fitness']:.8f}")
        print(f"Solution: x1 = {best_strategy[1]['best_solution'][0]:.8f}, "
              f"x2 = {best_strategy[1]['best_solution'][1]:.8f}")
        print(f"Error from theoretical optimum: {best_strategy[1]['error']:.8f}")
        print(f"Computation time: {best_strategy[1]['computation_time']:.4f}s")
        
        # Display all results in a table
        print("\n" + "-" * 70)
        print("ALL STRATEGIES PERFORMANCE:")
        print("-" * 70)
        for name, data in results.items():
            print(f"{name}: Fitness = {data['best_fitness']:.6f}, "
                  f"Time = {data['computation_time']:.3f}s, "
                  f"Error = {data['error']:.6f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)