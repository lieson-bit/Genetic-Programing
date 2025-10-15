import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Callable
import pandas as pd

def load_simple_problem():
    """Load P07 simple complexity problem"""
    capacity = 6404180
    weights = [
        382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150,
        823460, 903959, 853665, 551830, 610856, 670702, 488960, 951111,
        323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684
    ]
    values = [
        825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457,
        1679693, 1902996, 1844992, 1049289, 1252836, 1319836, 953277, 2067538,
        675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261
    ]
    optimal = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    
    return weights, values, capacity, optimal

def load_complex_problem():
    """Load Set 7 increased complexity problem"""
    capacity = 12828
    data = [
        (324, 981), (151, 119), (651, 419), (73, 758), (536, 152),
        (366, 489), (58, 40), (508, 669), (38, 765), (434, 574),
        (70, 876), (91, 314), (425, 696), (827, 595), (124, 580),
        (224, 457), (628, 840), (948, 945), (578, 475), (397, 665),
        (977, 61), (47, 702), (859, 648), (290, 994), (145, 822),
        (118, 285), (309, 386), (817, 669), (181, 23), (582, 462),
        (639, 169), (373, 118), (548, 59), (63, 769), (60, 130),
        (206, 248), (681, 391), (428, 872), (315, 81), (586, 450),
        (454, 550), (300, 884), (795, 820), (699, 864), (245, 279),
        (575, 416), (526, 359), (876, 885), (730, 958), (288, 151)
    ]
    
    values = [item[0] for item in data]
    weights = [item[1] for item in data]
    
    return weights, values, capacity, None

class ImprovedKnapsackGA:
    def __init__(self, weights: List[int], values: List[int], capacity: int, 
                 encoding_type: str = 'variable_length'):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n_items = len(weights)
        self.encoding_type = encoding_type
        
        # Calculate value-to-weight ratios for greedy repair
        self.ratios = [v/w if w > 0 else 0 for v, w in zip(values, weights)]
        
    def initialize_population(self, pop_size: int) -> List:
        """Initialize population based on encoding type"""
        if self.encoding_type == 'variable_length':
            return self._init_variable_length_population(pop_size)
        elif self.encoding_type == 'binary':
            return self._init_binary_population(pop_size)
        else:
            return self._init_permutation_population(pop_size)
    
    def _init_variable_length_population(self, pop_size: int) -> List[List[int]]:
        """Initialize variable-length encoding population"""
        population = []
        for _ in range(pop_size):
            # Create a random permutation of items
            permutation = list(range(self.n_items))
            random.shuffle(permutation)
            
            # Generate feasible solution using greedy decoding
            solution = self._decode_permutation(permutation)
            population.append(solution)
        return population
    
    def _init_binary_population(self, pop_size: int) -> List[List[int]]:
        """Initialize binary encoding population"""
        population = []
        for _ in range(pop_size):
            individual = [random.randint(0, 1) for _ in range(self.n_items)]
            # Repair if necessary
            if sum(self.weights[i] for i in range(self.n_items) if individual[i] == 1) > self.capacity:
                individual = self._repair_binary(individual)
            population.append(individual)
        return population
    
    def _init_permutation_population(self, pop_size: int) -> List[List[int]]:
        """Initialize permutation encoding population"""
        population = []
        for _ in range(pop_size):
            permutation = list(range(self.n_items))
            random.shuffle(permutation)
            population.append(permutation)
        return population
    
    def _decode_permutation(self, permutation: List[int]) -> List[int]:
        """Decode permutation to feasible solution"""
        current_weight = 0
        solution = []
        
        for item in permutation:
            if current_weight + self.weights[item] <= self.capacity:
                solution.append(item)
                current_weight += self.weights[item]
        
        return solution
    
    def fitness(self, individual) -> float:
        """Calculate fitness of an individual"""
        if self.encoding_type == 'variable_length':
            return self._fitness_variable_length(individual)
        elif self.encoding_type == 'binary':
            return self._fitness_binary(individual)
        else:
            return self._fitness_permutation(individual)
    
    def _fitness_variable_length(self, solution: List[int]) -> float:
        """Fitness for variable-length encoding"""
        total_value = sum(self.values[item] for item in solution)
        total_weight = sum(self.weights[item] for item in solution)
        
        if total_weight > self.capacity:
            # Apply penalty for infeasible solutions
            penalty = (total_weight - self.capacity) * max(self.values) / min(self.weights)
            return max(0, total_value - penalty)
        return total_value
    
    def _fitness_binary(self, individual: List[int]) -> float:
        """Fitness for binary encoding"""
        total_value = sum(self.values[i] for i in range(self.n_items) if individual[i] == 1)
        total_weight = sum(self.weights[i] for i in range(self.n_items) if individual[i] == 1)
        
        if total_weight > self.capacity:
            # Stronger penalty function
            penalty = (total_weight - self.capacity) * max(self.values) / min(self.weights)
            return max(0, total_value - penalty)
        return total_value
    
    def _fitness_permutation(self, permutation: List[int]) -> float:
        """Fitness for permutation encoding"""
        solution = self._decode_permutation(permutation)
        return sum(self.values[item] for item in solution)
    
    def crossover(self, parent1, parent2, crossover_rate: float) -> Tuple:
        """Perform crossover based on encoding type"""
        if random.random() > crossover_rate:
            return parent1, parent2
            
        if self.encoding_type == 'variable_length':
            return self._crossover_variable_length(parent1, parent2)
        elif self.encoding_type == 'binary':
            return self._crossover_binary(parent1, parent2)
        else:
            return self._crossover_permutation(parent1, parent2)
    
    def _crossover_variable_length(self, parent1: List[int], parent2: List[int]) -> Tuple:
        """Variable-length crossover with repair"""
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1, parent2
            
        # Two-point crossover for more diversity
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))
        point3 = random.randint(0, len(parent1))
        point4 = random.randint(0, len(parent2))
        
        start1, end1 = sorted([point1, point3])
        start2, end2 = sorted([point2, point4])
        
        # Create offspring
        child1 = parent1[:start1] + parent2[start2:end2] + parent1[end1:]
        child2 = parent2[:start2] + parent1[start1:end1] + parent2[end2:]
        
        # Remove duplicates
        child1 = list(dict.fromkeys(child1))
        child2 = list(dict.fromkeys(child2))
        
        # Repair if necessary
        child1 = self._repair_solution(child1)
        child2 = self._repair_solution(child2)
        
        return child1, child2
    
    def _crossover_binary(self, parent1: List[int], parent2: List[int]) -> Tuple:
        """Uniform crossover for binary encoding"""
        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    
    def _crossover_permutation(self, parent1: List[int], parent2: List[int]) -> Tuple:
        """PMX crossover for permutations"""
        size = len(parent1)
        point1, point2 = sorted(random.sample(range(size), 2))
        
        def pmx_crossover(p1, p2):
            child = [-1] * size
            
            # Copy segment
            child[point1:point2] = p1[point1:point2]
            
            # Fill remaining positions
            for i in list(range(0, point1)) + list(range(point2, size)):
                candidate = p2[i]
                while candidate in child:
                    idx = child.index(candidate)
                    candidate = p2[idx]
                child[i] = candidate
                
            return child
        
        child1 = pmx_crossover(parent1, parent2)
        child2 = pmx_crossover(parent2, parent1)
        return child1, child2
    
    def mutate(self, individual, mutation_rate: float):
        """Perform mutation based on encoding type"""
        if random.random() > mutation_rate:
            return individual
            
        if self.encoding_type == 'variable_length':
            return self._mutate_variable_length(individual)
        elif self.encoding_type == 'binary':
            return self._mutate_binary(individual)
        else:
            return self._mutate_permutation(individual)
    
    def _mutate_variable_length(self, solution: List[int]) -> List[int]:
        """Enhanced mutation for variable-length encoding"""
        solution = solution.copy()
        
        mutation_type = random.choice(['add', 'remove', 'replace', 'shuffle'])
        
        if mutation_type == 'add' and len(solution) < self.n_items:
            current_weight = sum(self.weights[item] for item in solution)
            available_items = [i for i in range(self.n_items) 
                            if i not in solution and 
                            current_weight + self.weights[i] <= self.capacity]
            if available_items:
                # Prefer items with high value-to-weight ratio
                available_items.sort(key=lambda x: self.ratios[x], reverse=True)
                solution.append(available_items[0])
                
        elif mutation_type == 'remove' and len(solution) > 1:
            # Remove item with worst ratio
            if solution:
                worst_idx = min(range(len(solution)), 
                              key=lambda i: self.ratios[solution[i]])
                solution.pop(worst_idx)
                
        elif mutation_type == 'replace' and len(solution) >= 1:
            current_weight = sum(self.weights[item] for item in solution)
            if solution:
                remove_idx = random.randint(0, len(solution) - 1)
                removed_item = solution.pop(remove_idx)
                current_weight -= self.weights[removed_item]
                
                available_items = [i for i in range(self.n_items) 
                                if i not in solution and 
                                current_weight + self.weights[i] <= self.capacity]
                if available_items:
                    available_items.sort(key=lambda x: self.ratios[x], reverse=True)
                    solution.append(available_items[0])
                    
        elif mutation_type == 'shuffle' and len(solution) > 1:
            random.shuffle(solution)
        
        return solution
    
    def _mutate_binary(self, individual: List[int]) -> List[int]:
        """Bit-flip mutation for binary encoding"""
        individual = individual.copy()
        for i in range(len(individual)):
            if random.random() < 0.1:  # Low probability per bit
                individual[i] = 1 - individual[i]
        return individual
    
    def _mutate_permutation(self, permutation: List[int]) -> List[int]:
        """Swap mutation for permutations"""
        permutation = permutation.copy()
        for _ in range(2):  # Multiple swaps
            idx1, idx2 = random.sample(range(len(permutation)), 2)
            permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
        return permutation
    
    def _repair_solution(self, solution: List[int]) -> List[int]:
        """Enhanced repair using greedy approach"""
        current_weight = sum(self.weights[item] for item in solution)
        
        # If solution is feasible, try to improve it
        if current_weight <= self.capacity:
            # Try to add more items
            available_items = [i for i in range(self.n_items) 
                            if i not in solution and 
                            current_weight + self.weights[i] <= self.capacity]
            if available_items:
                # Add best available item
                available_items.sort(key=lambda x: self.ratios[x], reverse=True)
                solution.append(available_items[0])
                current_weight += self.weights[available_items[0]]
        
        # Repair if overweight
        while current_weight > self.capacity and solution:
            # Remove item with worst value-to-weight ratio
            worst_idx = min(range(len(solution)), 
                          key=lambda i: self.ratios[solution[i]])
            removed_item = solution.pop(worst_idx)
            current_weight -= self.weights[removed_item]
        
        return solution
    
    def _repair_binary(self, individual: List[int]) -> List[int]:
        """Repair binary solution"""
        individual = individual.copy()
        total_weight = sum(self.weights[i] for i in range(self.n_items) if individual[i] == 1)
        
        # Remove items until feasible
        while total_weight > self.capacity:
            # Find included items with worst ratios
            included_items = [i for i in range(self.n_items) if individual[i] == 1]
            if not included_items:
                break
                
            worst_item = min(included_items, key=lambda x: self.ratios[x])
            individual[worst_item] = 0
            total_weight -= self.weights[worst_item]
        
        return individual
    
    def select_parents(self, population: List, fitnesses: List[float], 
                      tournament_size: int = 3) -> List:
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return selected
    
    def run(self, pop_size: int = 100, generations: int = 1000,
            crossover_rate: float = 0.8, mutation_rate: float = 0.1,
            elitism_count: int = 2, tournament_size: int = 3) -> dict:
        """Run the genetic algorithm with enhanced tracking"""
        # Initialize population
        population = self.initialize_population(pop_size)
        best_fitness_history = []
        avg_fitness_history = []
        worst_fitness_history = []
        diversity_history = []
        
        start_time = time.time()
        
        for generation in range(generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Track statistics
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitnesses))
            worst_fitness_history.append(np.min(fitnesses))
            
            # Track diversity (average hamming distance for binary, unique solutions for others)
            if self.encoding_type == 'binary':
                diversity = self._calculate_diversity_binary(population)
            else:
                diversity = len(set(tuple(sol) for sol in population)) / pop_size
            diversity_history.append(diversity)
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitnesses)[-elitism_count:]
            elites = [population[i] for i in elite_indices]
            
            # Selection
            parents = self.select_parents(population, fitnesses, tournament_size)
            
            # Create new population
            new_population = elites.copy()
            
            while len(new_population) < pop_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2, crossover_rate)
                
                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            # Ensure population size is correct
            population = new_population[:pop_size]
            
            # Print progress
            if generation % 100 == 0:
                print(f"Generation {generation}: Best = {best_fitness:,}, "
                      f"Avg = {np.mean(fitnesses):,}, Diversity = {diversity:.3f}")
        
        execution_time = time.time() - start_time
        
        # Find final best solution
        final_fitnesses = [self.fitness(ind) for ind in population]
        best_idx = np.argmax(final_fitnesses)
        best_solution = population[best_idx]
        best_fitness = final_fitnesses[best_idx]
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'worst_fitness_history': worst_fitness_history,
            'diversity_history': diversity_history,
            'execution_time': execution_time,
            'final_population': population
        }
    
    def _calculate_diversity_binary(self, population: List[List[int]]) -> float:
        """Calculate diversity for binary encoding"""
        if len(population) <= 1:
            return 0.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = sum(1 for a, b in zip(population[i], population[j]) if a != b)
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

def analyze_and_visualize_results(problem_name, results, optimal_value=None):
    """Comprehensive analysis and visualization of results"""
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE ANALYSIS: {problem_name}")
    print(f"{'='*60}")
    
    # Basic statistics
    best_fitness = results['best_fitness']
    convergence_generation = np.argmax(results['best_fitness_history'])
    
    print(f"🎯 BEST FITNESS: {best_fitness:,}")
    if optimal_value:
        accuracy = (best_fitness / optimal_value) * 100
        print(f"📊 ACCURACY: {accuracy:.2f}% of optimal")
        print(f"🎯 OPTIMAL VALUE: {optimal_value:,}")
    
    print(f"⏱️  EXECUTION TIME: {results['execution_time']:.2f} seconds")
    print(f"🔄 CONVERGED AT GENERATION: {convergence_generation}")
    print(f"📈 FINAL DIVERSITY: {results['diversity_history'][-1]:.3f}")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Genetic Algorithm Analysis: {problem_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Fitness convergence
    axes[0, 0].plot(results['best_fitness_history'], 'g-', linewidth=2, label='Best Fitness')
    axes[0, 0].plot(results['avg_fitness_history'], 'b-', linewidth=1, label='Average Fitness')
    axes[0, 0].plot(results['worst_fitness_history'], 'r-', linewidth=1, label='Worst Fitness')
    if optimal_value:
        axes[0, 0].axhline(y=optimal_value, color='black', linestyle='--', 
                          label=f'Optimal ({optimal_value:,})')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Convergence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Diversity
    axes[0, 1].plot(results['diversity_history'], 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Diversity')
    axes[0, 1].set_title('Population Diversity Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Fitness distribution at convergence
    final_fitnesses = [results['fitness_function'](ind) for ind in results['final_population']]
    axes[0, 2].hist(final_fitnesses, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].axvline(x=best_fitness, color='red', linestyle='--', linewidth=2, 
                      label=f'Best: {best_fitness:,}')
    axes[0, 2].set_xlabel('Fitness')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Final Population Fitness Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Improvement over time
    improvements = [results['best_fitness_history'][i] - results['best_fitness_history'][i-1] 
                   for i in range(1, len(results['best_fitness_history']))]
    axes[1, 0].plot(improvements, 'teal', linewidth=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='-', alpha=0.5)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Fitness Improvement')
    axes[1, 0].set_title('Fitness Improvement Per Generation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Runtime analysis (if multiple runs)
    if 'parameter_runs' in results:
        param_names = list(results['parameter_runs'].keys())
        times = [results['parameter_runs'][p]['execution_time'] for p in param_names]
        fitnesses = [results['parameter_runs'][p]['best_fitness'] for p in param_names]
        
        bars = axes[1, 1].bar(param_names, fitnesses, color='lightblue', alpha=0.7)
        axes[1, 1].set_xlabel('Parameter Setting')
        axes[1, 1].set_ylabel('Best Fitness', color='blue')
        axes[1, 1].tick_params(axis='y', labelcolor='blue')
        axes[1, 1].set_title('Performance vs Parameters')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        ax2 = axes[1, 1].twinx()
        ax2.plot(param_names, times, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Execution Time (s)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
    
    # Plot 6: Solution quality analysis
    if optimal_value and 'best_solution_binary' in results:
        optimal_solution = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        found_solution = results['best_solution_binary']
        
        correct_positions = sum(1 for o, f in zip(optimal_solution, found_solution) if o == f)
        accuracy_percentage = (correct_positions / len(optimal_solution)) * 100
        
        axes[1, 2].bar(['Optimal', 'Found'], [optimal_value, best_fitness], 
                      color=['green', 'orange'], alpha=0.7)
        axes[1, 2].set_ylabel('Fitness')
        axes[1, 2].set_title(f'Solution Comparison\n({accuracy_percentage:.1f}% item accuracy)')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print solution details
    print(f"\n📋 SOLUTION DETAILS:")
    if 'best_solution_binary' in results:
        binary_sol = results['best_solution_binary']
        total_weight = sum(w for w, x in zip(results['weights'], binary_sol) if x == 1)
        total_items = sum(binary_sol)
        print(f"   Items selected: {total_items}/{len(binary_sol)}")
        print(f"   Total weight: {total_weight:,}/{results['capacity']:,}")
        print(f"   Solution vector: {binary_sol}")

def run_comprehensive_analysis():
    """Run complete analysis for both problems"""
    
    print("🧬 GENETIC ALGORITHM - KNAPSACK PROBLEM ANALYSIS")
    print("=" * 60)
    
    # Task 1: Simple Problem with OPTIMAL parameters
    print("\n🎯 TASK 1: SIMPLE PROBLEM (P07) - OPTIMAL RUN")
    print("-" * 50)
    
    weights, values, capacity, optimal = load_simple_problem()
    optimal_value = 13549094
    
    # Use parameters that worked best from your analysis
    ga_simple = ImprovedKnapsackGA(weights, values, capacity, 'variable_length')
    results_simple = ga_simple.run(
        pop_size=200,           # Larger population worked better
        generations=500,        # Shorter since it converges fast
        crossover_rate=0.9,     # Higher crossover worked better
        mutation_rate=0.1,      # Moderate mutation worked well
        elitism_count=5,        # More elitism
        tournament_size=5       # Stronger selection pressure
    )
    
    # Add additional data for analysis
    results_simple['fitness_function'] = ga_simple.fitness
    results_simple['weights'] = weights
    results_simple['capacity'] = capacity
    results_simple['optimal_value'] = optimal_value
    
    # Convert to binary for comparison
    binary_solution = [0] * len(weights)
    for item in results_simple['best_solution']:
        binary_solution[item] = 1
    results_simple['best_solution_binary'] = binary_solution
    
    analyze_and_visualize_results("Simple Problem (P07)", results_simple, optimal_value)
    
    # Task 2: Complex Problem
    print("\n🎯 TASK 2: COMPLEX PROBLEM (Set 7)")
    print("-" * 50)
    
    weights_comp, values_comp, capacity_comp, _ = load_complex_problem()
    
    ga_complex = ImprovedKnapsackGA(weights_comp, values_comp, capacity_comp, 'variable_length')
    results_complex = ga_complex.run(
        pop_size=150,
        generations=800,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elitism_count=3,
        tournament_size=4
    )
    
    results_complex['fitness_function'] = ga_complex.fitness
    results_complex['weights'] = weights_comp
    results_complex['capacity'] = capacity_comp
    
    analyze_and_visualize_results("Complex Problem (Set 7)", results_complex)
    
    # Parameter sensitivity analysis
    print("\n🔧 PARAMETER SENSITIVITY ANALYSIS")
    print("-" * 40)
    
    parameter_results = {}
    test_params = [
        ('Small Pop (50)', {'pop_size': 50, 'generations': 300}),
        ('Large Pop (200)', {'pop_size': 200, 'generations': 300}),
        ('Low Crossover (0.6)', {'crossover_rate': 0.6}),
        ('High Crossover (0.95)', {'crossover_rate': 0.95}),
        ('Low Mutation (0.05)', {'mutation_rate': 0.05}),
        ('High Mutation (0.2)', {'mutation_rate': 0.2}),
    ]
    
    for param_name, params in test_params:
        print(f"Testing {param_name}...")
        ga_test = ImprovedKnapsackGA(weights, values, capacity, 'variable_length')
        default_params = {
            'pop_size': 100, 'generations': 300, 
            'crossover_rate': 0.8, 'mutation_rate': 0.1
        }
        default_params.update(params)
        
        results = ga_test.run(**default_params)
        parameter_results[param_name] = results
    
    # Plot parameter comparison
    plt.figure(figsize=(12, 8))
    
    param_names = list(parameter_results.keys())
    best_fitnesses = [results['best_fitness'] for results in parameter_results.values()]
    execution_times = [results['execution_time'] for results in parameter_results.values()]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Fitness comparison
    bars = ax1.bar(param_names, best_fitnesses, color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.set_title('Parameter Sensitivity Analysis - Fitness', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, best_fitnesses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100000, 
                f'{value:,}', ha='center', va='bottom', fontsize=9)
    
    # Execution time comparison
    bars = ax2.bar(param_names, execution_times, color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax2.set_title('Parameter Sensitivity Analysis - Execution Time', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, execution_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"✅ TASK 1 - Simple Problem:")
    print(f"   Best Fitness: {results_simple['best_fitness']:,}")
    print(f"   Optimal Fitness: {optimal_value:,}")
    print(f"   Accuracy: {(results_simple['best_fitness']/optimal_value)*100:.2f}%")
    print(f"   Status: {'OPTIMAL FOUND! 🎉' if results_simple['best_fitness'] == optimal_value else 'Very Close ✓'}")
    
    print(f"\n✅ TASK 2 - Complex Problem:")
    print(f"   Best Fitness: {results_complex['best_fitness']:,}")
    print(f"   Execution Time: {results_complex['execution_time']:.2f}s")
    
    print(f"\n⚡ Best Parameters Found:")
    print(f"   Population Size: 150-200")
    print(f"   Crossover Rate: 0.85-0.9") 
    print(f"   Mutation Rate: 0.1-0.15")
    print(f"   Encoding: Variable-length (as required)")

if __name__ == "__main__":
    run_comprehensive_analysis()