import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Callable

class KnapsackGA:
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
            penalty = (total_weight - self.capacity) * max(self.ratios)
            return total_value - penalty
        return total_value
    
    def _fitness_binary(self, individual: List[int]) -> float:
        """Fitness for binary encoding"""
        total_value = sum(self.values[i] for i in range(self.n_items) if individual[i] == 1)
        total_weight = sum(self.weights[i] for i in range(self.n_items) if individual[i] == 1)
        
        if total_weight > self.capacity:
            # Penalty function
            penalty = (total_weight - self.capacity) * max(self.ratios)
            return total_value - penalty
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
            
        # Single-point crossover
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))
        
        # Create offspring
        child1 = parent1[:point1] + parent2[point2:]
        child2 = parent2[:point2] + parent1[point1:]
        
        # Remove duplicates
        child1 = list(dict.fromkeys(child1))
        child2 = list(dict.fromkeys(child2))
        
        # Repair if necessary
        child1 = self._repair_solution(child1)
        child2 = self._repair_solution(child2)
        
        return child1, child2
    
    def _crossover_binary(self, parent1: List[int], parent2: List[int]) -> Tuple:
        """Single-point binary crossover"""
        point = random.randint(1, self.n_items - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def _crossover_permutation(self, parent1: List[int], parent2: List[int]) -> Tuple:
        """Order crossover for permutations"""
        size = len(parent1)
        point1, point2 = sorted(random.sample(range(size), 2))
        
        def create_child(p1, p2):
            child = [None] * size
            # Copy segment from parent1
            child[point1:point2] = p1[point1:point2]
            # Fill remaining positions with items from parent2
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while p2[pointer] in child:
                        pointer += 1
                    child[i] = p2[pointer]
            return child
        
        child1 = create_child(parent1, parent2)
        child2 = create_child(parent2, parent1)
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
        """Mutation for variable-length encoding"""
        if len(solution) == 0:
            return solution
            
        # Randomly remove an item
        if len(solution) > 1 and random.random() < 0.5:
            solution = solution.copy()
            remove_idx = random.randint(0, len(solution) - 1)
            solution.pop(remove_idx)
        
        # Randomly add a feasible item
        if random.random() < 0.5:
            current_weight = sum(self.weights[item] for item in solution)
            available_items = [i for i in range(self.n_items) 
                            if i not in solution and 
                            current_weight + self.weights[i] <= self.capacity]
            if available_items:
                solution = solution.copy()
                solution.append(random.choice(available_items))
        
        return solution
    
    def _mutate_binary(self, individual: List[int]) -> List[int]:
        """Bit-flip mutation for binary encoding"""
        individual = individual.copy()
        idx = random.randint(0, self.n_items - 1)
        individual[idx] = 1 - individual[idx]
        return individual
    
    def _mutate_permutation(self, permutation: List[int]) -> List[int]:
        """Swap mutation for permutations"""
        permutation = permutation.copy()
        idx1, idx2 = random.sample(range(len(permutation)), 2)
        permutation[idx1], permutation[idx2] = permutation[idx2], permutation[idx1]
        return permutation
    
    def _repair_solution(self, solution: List[int]) -> List[int]:
        """Repair infeasible solution using greedy approach"""
        current_weight = sum(self.weights[item] for item in solution)
        
        while current_weight > self.capacity and solution:
            # Find item with minimum value-to-weight ratio
            min_ratio_idx = min(range(len(solution)), 
                              key=lambda i: self.ratios[solution[i]])
            removed_item = solution.pop(min_ratio_idx)
            current_weight -= self.weights[removed_item]
        
        return solution
    
    def select_parents(self, population: List, fitnesses: List[float], 
                      tournament_size: int = 3) -> List:
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
            winner = max(tournament, key=lambda x: x[1])[0]
            selected.append(winner)
        return selected
    
    def run(self, pop_size: int = 100, generations: int = 1000,
            crossover_rate: float = 0.8, mutation_rate: float = 0.1,
            elitism_count: int = 2, tournament_size: int = 3) -> dict:
        """Run the genetic algorithm"""
        # Initialize population
        population = self.initialize_population(pop_size)
        best_fitness_history = []
        avg_fitness_history = []
        convergence_data = []
        
        start_time = time.time()
        
        for generation in range(generations):
            # Evaluate fitness
            fitnesses = [self.fitness(ind) for ind in population]
            
            # Track best solution
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitnesses))
            
            # Store convergence data
            convergence_data.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitnesses),
                'best_solution': best_solution.copy()
            })
            
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
                print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")
        
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
            'convergence_data': convergence_data,
            'execution_time': execution_time
        }

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

def plot_convergence(results_dict, title):
    """Plot convergence graphs"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    for label, results in results_dict.items():
        plt.plot(results['best_fitness_history'], label=label)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title(f'{title} - Best Fitness Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    for label, results in results_dict.items():
        plt.plot(results['avg_fitness_history'], label=label)
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title(f'{title} - Average Fitness Convergence')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    execution_times = [results['execution_time'] for results in results_dict.values()]
    labels = list(results_dict.keys())
    plt.bar(labels, execution_times)
    plt.xlabel('Parameter Setting')
    plt.ylabel('Execution Time (s)')
    plt.title(f'{title} - Execution Time')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_parameters(weights, values, capacity, encoding_type, problem_name):
    """Analyze different parameter settings"""
    param_results = {}
    
    # Test different population sizes
    print(f"Testing population sizes for {problem_name}...")
    pop_sizes = [50, 100, 200]
    for pop_size in pop_sizes:
        ga = KnapsackGA(weights, values, capacity, encoding_type)
        results = ga.run(pop_size=pop_size, generations=500, 
                        crossover_rate=0.8, mutation_rate=0.1)
        param_results[f'PopSize_{pop_size}'] = results
    
    # Test different crossover rates
    print(f"Testing crossover rates for {problem_name}...")
    crossover_rates = [0.6, 0.8, 0.9]
    for cr in crossover_rates:
        ga = KnapsackGA(weights, values, capacity, encoding_type)
        results = ga.run(pop_size=100, generations=500, 
                        crossover_rate=cr, mutation_rate=0.1)
        param_results[f'Crossover_{cr}'] = results
    
    # Test different mutation rates
    print(f"Testing mutation rates for {problem_name}...")
    mutation_rates = [0.05, 0.1, 0.2]
    for mr in mutation_rates:
        ga = KnapsackGA(weights, values, capacity, encoding_type)
        results = ga.run(pop_size=100, generations=500, 
                        crossover_rate=0.8, mutation_rate=mr)
        param_results[f'Mutation_{mr}'] = results
    
    return param_results

def main():
    print("Knapsack Problem Solution using Genetic Algorithms")
    print("=" * 50)
    
    # Task 1: Simple Complexity Problem
    print("\nTASK 1: Simple Complexity Problem (P07)")
    print("-" * 40)
    
    weights, values, capacity, optimal = load_simple_problem()
    
    # Run GA for simple problem
    ga_simple = KnapsackGA(weights, values, capacity, 'variable_length')
    results_simple = ga_simple.run(pop_size=100, generations=1000, 
                                  crossover_rate=0.8, mutation_rate=0.1)
    
    print(f"Best Fitness Found: {results_simple['best_fitness']:,}")
    print(f"Optimal Fitness: 13,549,094")
    print(f"Percentage of Optimal: {(results_simple['best_fitness']/13549094)*100:.2f}%")
    print(f"Execution Time: {results_simple['execution_time']:.2f} seconds")
    
    # Convert best solution to binary format for comparison
    if ga_simple.encoding_type == 'variable_length':
        binary_solution = [0] * len(weights)
        for item in results_simple['best_solution']:
            binary_solution[item] = 1
        print(f"Best Solution: {binary_solution}")
        print(f"Optimal Solution: {optimal}")
    
    # Parameter analysis for simple problem
    print("\nAnalyzing parameters for simple problem...")
    param_results_simple = analyze_parameters(weights, values, capacity, 
                                            'variable_length', 'Simple Problem')
    plot_convergence(param_results_simple, 'Simple Problem Parameter Analysis')
    
    # Task 2: Increased Complexity Problem
    print("\nTASK 2: Increased Complexity Problem (Set 7)")
    print("-" * 40)
    
    weights_complex, values_complex, capacity_complex, _ = load_complex_problem()
    
    # Run GA for complex problem
    ga_complex = KnapsackGA(weights_complex, values_complex, capacity_complex, 'variable_length')
    results_complex = ga_complex.run(pop_size=150, generations=1500, 
                                   crossover_rate=0.8, mutation_rate=0.1)
    
    print(f"Best Fitness Found: {results_complex['best_fitness']:,}")
    print(f"Execution Time: {results_complex['execution_time']:.2f} seconds")
    
    # Parameter analysis for complex problem
    print("\nAnalyzing parameters for complex problem...")
    param_results_complex = analyze_parameters(weights_complex, values_complex, capacity_complex,
                                             'variable_length', 'Complex Problem')
    plot_convergence(param_results_complex, 'Complex Problem Parameter Analysis')
    
    # Comparative analysis
    print("\nCOMPARATIVE ANALYSIS")
    print("-" * 40)
    
    # Test different encoding types for comparison
    encoding_types = ['variable_length', 'binary', 'permutation']
    encoding_results = {}
    
    for encoding in encoding_types:
        print(f"Testing {encoding} encoding...")
        ga_test = KnapsackGA(weights_complex, values_complex, capacity_complex, encoding)
        results = ga_test.run(pop_size=100, generations=500)
        encoding_results[encoding] = results
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for encoding, results in encoding_results.items():
        plt.plot(results['best_fitness_history'][:200], label=encoding)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Encoding Type Comparison - Best Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    times = [results['execution_time'] for results in encoding_results.values()]
    plt.bar(encoding_results.keys(), times)
    plt.xlabel('Encoding Type')
    plt.ylabel('Execution Time (s)')
    plt.title('Encoding Type Comparison - Execution Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nSUMMARY")
    print("-" * 20)
    print(f"Simple Problem (P07):")
    print(f"  Best Solution: {results_simple['best_fitness']:,}")
    print(f"  Optimal Solution: 13,549,094")
    print(f"  Accuracy: {(results_simple['best_fitness']/13549094)*100:.2f}%")
    
    print(f"\nComplex Problem (Set 7):")
    print(f"  Best Solution: {results_complex['best_fitness']:,}")
    
    print(f"\nBest Performing Parameters:")
    best_simple = max(param_results_simple.items(), key=lambda x: x[1]['best_fitness'])
    best_complex = max(param_results_complex.items(), key=lambda x: x[1]['best_fitness'])
    print(f"  Simple Problem: {best_simple[0]} (Fitness: {best_simple[1]['best_fitness']:,})")
    print(f"  Complex Problem: {best_complex[0]} (Fitness: {best_complex[1]['best_fitness']:,})")

if __name__ == "__main__":
    main()