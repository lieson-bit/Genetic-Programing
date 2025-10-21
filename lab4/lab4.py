import numpy as np
import random
import operator
import math
from deap import base, creator, tools, gp
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class SymbolicRegressionGP:
    def __init__(self):
        self.target_function = self.define_target_function()
        self.pset = self.create_primitive_set()
        self.toolbox = self.create_toolbox()
        self.setup_evolution()
        
    def define_target_function(self):
        """Target function: 30*x*z/((x-10)*y**2)"""
        def target_func(x, y, z):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = 30 * x * z / ((x - 10) * y**2)
                result = np.nan_to_num(result, nan=0.0, posinf=1000, neginf=-1000)
            return result
        return target_func

    def protected_div(self, a, b):
        """Protected division - avoid division by zero"""
        if abs(b) < 1e-10:
            return 1.0
        return a / b

    def protected_pow(self, a, b):
        """Protected power - avoid complex numbers and overflow"""
        try:
            if a < 0 and not float(b).is_integer():
                return 1.0
            result = a ** b
            if isinstance(result, complex):
                return 1.0
            if math.isnan(result) or math.isinf(result):
                return 1.0
            return result
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1.0

    def create_primitive_set(self):
        """Create the primitive set for GP"""
        pset = gp.PrimitiveSet("MAIN", 3)
        pset.renameArguments(ARG0='x', ARG1='y', ARG2='z')

        # Core arithmetic functions
        pset.addPrimitive(operator.add, 2, name="add")
        pset.addPrimitive(operator.sub, 2, name="sub")
        pset.addPrimitive(operator.mul, 2, name="mul")
        pset.addPrimitive(self.protected_div, 2, name="div")
        pset.addPrimitive(operator.neg, 1, name="neg")
        
        # Mathematical functions
        pset.addPrimitive(math.sin, 1, name="sin")
        pset.addPrimitive(math.cos, 1, name="cos")
        pset.addPrimitive(math.exp, 1, name="exp")
        pset.addPrimitive(self.protected_pow, 2, name="pow")
        pset.addPrimitive(abs, 1, name="abs")

        # Constants and terminals - ADD MORE RELEVANT CONSTANTS
        pset.addEphemeralConstant("rand_int", partial(random.randint, -10, 10))
        pset.addEphemeralConstant("rand_float", partial(random.uniform, -10, 10))
        
        # Useful constants for our target function
        pset.addTerminal(1.0, name="one")
        pset.addTerminal(2.0, name="two")
        pset.addTerminal(10.0, name="ten")
        pset.addTerminal(30.0, name="thirty")
        pset.addTerminal(-10.0, name="neg_ten")
        pset.addTerminal(0.0, name="zero")
        pset.addTerminal(-1.0, name="neg_one")

        return pset

    def eval_symreg(self, individual, points, targets):
        """Evaluation function for symbolic regression"""
        try:
            func = self.toolbox.compile(expr=individual)
        except:
            return (10000.0,)

        predictions = []
        for x, y, z in points:
            try:
                pred = func(x, y, z)
                if (isinstance(pred, complex) or math.isnan(pred) or 
                    math.isinf(pred) or abs(pred) > 1e10):
                    pred = 1000.0
                predictions.append(float(pred))
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                predictions.append(1000.0)

        try:
            mse = np.mean((np.array(predictions) - targets) ** 2)
            return (mse,)
        except:
            return (10000.0,)

    def create_toolbox(self):
        """Create and configure the DEAP toolbox"""
        # Create types
        if "FitnessMin" not in creator.__dict__:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        return toolbox

    def setup_evolution(self):
        """Setup evolutionary parameters and operators"""
        # IMPROVED PARAMETERS
        self.params = {
            'population_size': 1000,  # Increased population
            'crossover_prob': 0.85,   # Adjusted probabilities
            'mutation_prob': 0.15,    # Higher mutation for exploration
            'generations': 150,       # More generations
            'tournament_size': 3,
            'training_points': 200,   # More training data
            'test_points': 50,
            'elite_size': 10          # Elite preservation
        }

        # Generate training data with BETTER DOMAIN
        self.points, self.targets = self.generate_training_data(self.params['training_points'])
        
        # Register operators
        self.toolbox.register("evaluate", self.eval_symreg, points=self.points, targets=self.targets)
        self.toolbox.register("select", tools.selTournament, tournsize=self.params['tournament_size'])
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=self.pset)

        # Bloat control - RELAXED constraints
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=15))

    def generate_training_data(self, n_samples):
        """Generate training data with BETTER domain handling"""
        # Use a more appropriate domain that avoids singularities
        x = np.random.uniform(-5, 5, n_samples)   # Wider range but avoid x=10
        y = np.random.uniform(0.5, 5, n_samples)   # Avoid y near 0
        z = np.random.uniform(-5, 5, n_samples)
        
        # Ensure no problematic values
        x = np.where(np.abs(x - 10) < 2, 8.0, x)  # Keep x away from 10
        y = np.where(y < 0.5, 0.5, y)             # Ensure y >= 0.5
        
        targets = self.target_function(x, y, z)
        return list(zip(x, y, z)), targets

    def tree_to_latex(self, tree):
        """Convert tree to more readable format"""
        return str(tree).replace('add', '+').replace('sub', '-').replace('mul', '*').replace('div', '/')

    def plot_3d_comparison(self, gp_func, generation, best_fitness):
        """Create 3D comparison plot between target and evolved function"""
        fig = plt.figure(figsize=(18, 6))
        
        # Create meshgrid for visualization with BETTER domain
        x_vals = np.linspace(-5, 5, 25)
        y_vals = np.linspace(0.5, 5, 25)  # Avoid y=0
        z_val = 1.0  # Fixed z for 2D slice
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Calculate target function
        Z_target = self.target_function(X, Y, z_val)
        
        # Calculate GP function
        Z_gp = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    pred = gp_func(X[i,j], Y[i,j], z_val)
                    if math.isnan(pred) or math.isinf(pred):
                        pred = 0
                    Z_gp[i,j] = pred
                except:
                    Z_gp[i,j] = 0

        # Plot 1: Target function
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z_target, cmap='viridis', alpha=0.8)
        ax1.set_title('Target Function\n$f(x,y,z) = \\frac{30xz}{(x-10)y^2}$', fontsize=14, pad=20)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(x,y,1.0)')
        ax1.view_init(30, 45)

        # Plot 2: GP function
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, Z_gp, cmap='plasma', alpha=0.8)
        ax2.set_title(f'GP Function (Gen {generation})\nMSE: {best_fitness:.4f}', fontsize=14, pad=20)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('f(x,y,1.0)')
        ax2.view_init(30, 45)

        # Plot 3: Error surface
        ax3 = fig.add_subplot(133, projection='3d')
        error = np.abs(Z_target - Z_gp)
        surf3 = ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
        ax3.set_title('Absolute Error', fontsize=14, pad=20)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('|Error|')
        ax3.view_init(30, 45)

        plt.tight_layout()
        plt.savefig(f'3d_comparison_gen_{generation}.png', dpi=150, bbox_inches='tight')  # Lower DPI
        plt.close()

    def plot_fitness_progression(self, logbook):
        """Enhanced fitness progression plot"""
        gens = logbook.select("gen")
        fit_mins = logbook.chapters["fitness"].select("min")
        fit_avgs = logbook.chapters["fitness"].select("avg")
        fit_stds = logbook.chapters["fitness"].select("std")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Smaller figure

        # Plot 1: Fitness progression
        ax1.plot(gens, fit_mins, 'b-', linewidth=2, label='Best Fitness')
        ax1.plot(gens, fit_avgs, 'r-', linewidth=2, label='Average Fitness')
        ax1.fill_between(gens, np.array(fit_avgs)-np.array(fit_stds), 
                        np.array(fit_avgs)+np.array(fit_stds), alpha=0.2, color='red')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness (MSE)')
        ax1.set_title('Fitness Progression', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Plot 2: Improvement percentage
        improvement = [(1 - fit/fit_mins[0]) * 100 for fit in fit_mins]
        ax2.plot(gens, improvement, 'g-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Improvement from Initial', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.fill_between(gens, improvement, alpha=0.3, color='green')

        final_improvement = improvement[-1]
        ax2.axhline(y=final_improvement, color='red', linestyle='--', 
                   label=f'Final: {final_improvement:.1f}%')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('fitness_progression.png', dpi=150, bbox_inches='tight')  # Lower DPI
        plt.close()

    def plot_formula_complexity(self, best_individuals):
        """Plot formula complexity vs fitness"""
        generations = [gen for gen, _, _ in best_individuals]
        complexities = [len(ind) for _, ind, _ in best_individuals]
        fitnesses = [fit for _, _, fit in best_individuals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Smaller figure

        # Complexity progression
        ax1.plot(generations, complexities, 'purple', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Formula Complexity (Nodes)')
        ax1.set_title('Formula Complexity', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Complexity vs Fitness
        scatter = ax2.scatter(complexities, fitnesses, c=generations, cmap='viridis', 
                             alpha=0.6, s=50)
        ax2.set_xlabel('Formula Complexity (Nodes)')
        ax2.set_ylabel('Fitness (MSE)')
        ax2.set_title('Complexity vs Fitness', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Generation')

        plt.tight_layout()
        plt.savefig('complexity_analysis.png', dpi=150, bbox_inches='tight')  # Lower DPI
        plt.close()

    def create_summary_report(self, best_individual, best_history, logbook):
        """Create a SIMPLIFIED summary report to avoid size issues"""
        best_func = self.toolbox.compile(expr=best_individual)
        
        # Test on new data
        test_points, test_targets = self.generate_training_data(self.params['test_points'])
        test_predictions = []
        
        for x, y, z in test_points:
            try:
                pred = best_func(x, y, z)
                if (isinstance(pred, complex) or math.isnan(pred) or math.isinf(pred)):
                    pred = 0.0
                test_predictions.append(float(pred))
            except:
                test_predictions.append(0.0)
        
        test_mse = np.mean((np.array(test_predictions) - test_targets) ** 2)
        train_mse = best_individual.fitness.values[0]
        
        # Create SIMPLIFIED summary figure
        fig = plt.figure(figsize=(10, 8))  # MUCH smaller figure
        
        # Main title
        fig.suptitle('Symbolic Regression Summary\n'
                    f'Target: $f(x,y,z) = \\frac{{30xz}}{{(x-10)y^2}}$', 
                    fontsize=14, fontweight='bold', y=0.98)

        # Simple text summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = (
            f"FINAL RESULTS:\n"
            f"• Training MSE: {train_mse:.6f}\n"
            f"• Test MSE: {test_mse:.6f}\n"
            f"• Improvement: {(1 - train_mse/best_history[0][2]) * 100:.1f}%\n"
            f"• Formula Complexity: {len(best_individual)} nodes\n"
            f"• Overfitting Ratio: {test_mse/train_mse:.2f}\n\n"
            f"BEST FORMULA:\n{self.tree_to_latex(best_individual)}\n\n"
            f"KEY PARAMETERS:\n"
            f"• Population: {self.params['population_size']}\n"
            f"• Generations: {self.params['generations']}\n"
            f"• Crossover: {self.params['crossover_prob']}\n"
            f"• Mutation: {self.params['mutation_prob']}"
        )
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
                family='monospace')

        plt.tight_layout()
        plt.savefig('summary_report.png', dpi=150, bbox_inches='tight')  # Lower DPI
        plt.close()

    def run_evolution(self):
        """Main evolutionary process with ELITISM"""
        print("🚀 Starting Genetic Programming for Symbolic Regression")
        print("=" * 60)
        
        # Initialize population
        pop = self.toolbox.population(n=self.params['population_size'])
        
        # Evaluate initial population
        print("📊 Evaluating initial population...")
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Statistics setup
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])

        # Record initial statistics
        record = mstats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        print(f"📍 Generation 0: Best MSE = {record['fitness']['min']:.6f}")

        best_individuals = []
        best_individuals.append((0, tools.selBest(pop, 1)[0], record['fitness']['min']))

        # Evolutionary loop with ELITISM
        print("🔄 Starting evolutionary process...")
        for gen in range(1, self.params['generations'] + 1):
            # Select elite individuals
            elite = tools.selBest(pop, self.params['elite_size'])
            
            # Selection and variation
            offspring = self.toolbox.select(pop, len(pop) - self.params['elite_size'])
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.params['crossover_prob']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.params['mutation_prob']:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Create new population: elite + offspring
            pop[:] = elite + offspring

            # Statistics
            record = mstats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Store best individual
            best_ind = tools.selBest(pop, 1)[0]
            best_individuals.append((gen, best_ind, best_ind.fitness.values[0]))

            # Progress reporting
            if gen % 10 == 0 or gen <= 5:
                improvement = (1 - best_ind.fitness.values[0] / best_individuals[0][2]) * 100
                print(f"📍 Generation {gen:3d}: Best MSE = {best_ind.fitness.values[0]:.6f} "
                      f"| Improvement = {improvement:5.1f}%")

            # Visualization at key generations
            if gen % 25 == 0 or gen <= 5:
                try:
                    gp_func = self.toolbox.compile(expr=best_ind)
                    self.plot_3d_comparison(gp_func, gen, best_ind.fitness.values[0])
                except Exception as e:
                    print(f"⚠️  Could not create 3D plot for generation {gen}: {e}")

        # Final results and visualization
        best_ind = tools.selBest(pop, 1)[0]
        
        print("\n" + "=" * 60)
        print("🎯 EVOLUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Create all final visualizations
        try:
            self.plot_fitness_progression(logbook)
            self.plot_formula_complexity(best_individuals)
            self.create_summary_report(best_ind, best_individuals, logbook)
            
            # Final 3D comparison
            gp_func = self.toolbox.compile(expr=best_ind)
            self.plot_3d_comparison(gp_func, self.params['generations'], best_ind.fitness.values[0])
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")

        return best_ind, best_individuals, logbook

def main():
    """Main execution function"""
    try:
        # Initialize GP system
        gp_system = SymbolicRegressionGP()
        
        # Run evolution
        best_individual, best_history, stats_log = gp_system.run_evolution()
        
        # Final results
        final_fitness = best_individual.fitness.values[0]
        initial_fitness = best_history[0][2]
        improvement = (1 - final_fitness / initial_fitness) * 100
        
        print(f"\n📈 FINAL RESULTS:")
        print(f"   • Initial MSE: {initial_fitness:.6f}")
        print(f"   • Final MSE: {final_fitness:.6f}")
        print(f"   • Total Improvement: {improvement:.1f}%")
        print(f"   • Best Formula: {gp_system.tree_to_latex(best_individual)}")
        print(f"\n💾 Results saved as:")
        print("   • summary_report.png - Comprehensive analysis")
        print("   • fitness_progression.png - Evolution progress")
        print("   • complexity_analysis.png - Formula complexity")
        print("   • 3d_comparison_gen_*.png - 3D function comparisons")
        
        # Show the actual target vs predicted values for a few points
        print(f"\n🔍 SAMPLE PREDICTIONS:")
        test_points = [(-2, 2, 1), (0, 1, 2), (3, 2, -1), (5, 3, 2)]
        best_func = gp_system.toolbox.compile(expr=best_individual)
        
        for x, y, z in test_points:
            target = gp_system.target_function(x, y, z)
            predicted = best_func(x, y, z)
            error = abs(target - predicted)
            print(f"   f({x}, {y}, {z}) | Target: {target:8.4f} | Pred: {predicted:8.4f} | Error: {error:8.4f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evolution interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()