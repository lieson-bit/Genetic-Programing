import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to maximize
def f(x):
    if np.any(x == 2):
        return -np.inf
    return np.cos(2*x) / np.abs(x - 2)

# GA parameters
POP_SIZE = 50
GENES = 16
GENERATIONS = 50
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01

# Domain intervals (split at singularity)
intervals = [(-10, 2-1e-5), (2+1e-5, 10)]

# Binary to real mapping
def binary_to_real(chrom, x_min, x_max):
    integer = int("".join(str(int(b)) for b in chrom), 2)
    max_int = 2**GENES - 1
    return x_min + (integer / max_int) * (x_max - x_min)

# Initialize population
def init_population():
    return np.random.randint(0, 2, size=(POP_SIZE, GENES))

# Evaluate fitness
def evaluate(pop, x_min, x_max):
    return np.array([f(binary_to_real(chrom, x_min, x_max)) for chrom in pop])

# Selection
def select(pop, fitness):
    total_fit = np.sum(fitness - np.min(fitness) + 1e-6)
    probs = (fitness - np.min(fitness) + 1e-6) / total_fit
    indices = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, p=probs)
    return pop[indices]

# Crossover
def crossover(pop):
    for i in range(0, POP_SIZE, 2):
        if np.random.rand() < CROSSOVER_RATE:
            point = np.random.randint(1, GENES)
            pop[i, point:], pop[i+1, point:] = pop[i+1, point:].copy(), pop[i, point:].copy()
    return pop

# Mutation
def mutate(pop):
    for i in range(POP_SIZE):
        for j in range(GENES):
            if np.random.rand() < MUTATION_RATE:
                pop[i, j] = 1 - pop[i, j]
    return pop

# Run GA with history for animation
def run_ga(x_min, x_max):
    pop = init_population()
    history = []
    best_x, best_f = None, -np.inf

    for gen in range(GENERATIONS):
        fitness = evaluate(pop, x_min, x_max)
        xs = np.array([binary_to_real(chrom, x_min, x_max) for chrom in pop])
        history.append((xs, fitness))

        # Track the best solution
        idx = np.argmax(fitness)
        if fitness[idx] > best_f:
            best_f = fitness[idx]
            best_x = xs[idx]

        pop = select(pop, fitness)
        pop = crossover(pop)
        pop = mutate(pop)

    return history, best_x, best_f

# Combine histories from both intervals and find overall best
history = []
best_global_x, best_global_f = None, -np.inf

for (x_min, x_max) in intervals:
    h, x, fx = run_ga(x_min, x_max)
    history += h
    if fx > best_global_f:
        best_global_f = fx
        best_global_x = x

print(f"\n✅ Global maximum found: f(x)={best_global_f:.5f} at x={best_global_x:.5f}")

# Plot setup
fig, ax = plt.subplots()
x_vals_left = np.linspace(-10, 2-1e-3, 1000)
x_vals_right = np.linspace(2+1e-3, 10, 1000)
ax.plot(x_vals_left, f(x_vals_left), 'r', label="f(x)")
ax.plot(x_vals_right, f(x_vals_right), 'r')

scat = ax.scatter([], [], c='b', s=20)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_title("GA optimization of f(x)=cos(2x)/|x-2|")

# Animation update
def update(frame):
    xs, fitness = history[frame]
    scat.set_offsets(np.c_[xs, fitness])
    ax.set_xlabel(f"Generation {frame+1}")
    return scat,

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=200, repeat=False)
plt.show()
