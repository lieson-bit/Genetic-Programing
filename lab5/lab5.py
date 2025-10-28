import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import pandas as pd

# Функция Гольдстейна-Прайса
def goldstein_price(x):
    x1, x2 = x
    term1 = (1 + (x1 + x2 + 1)**2 * 
            (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * 
            (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ТЕСТОВЫЕ ФУНКЦИИ ДЛЯ n=3 ТЕСТИРОВАНИЯ
# ============================================================================

def sphere_function(x):
    """Сферическая функция - работает для любого количества измерений"""
    return np.sum(np.array(x)**2)

def rastrigin_function(x):
    """Функция Растригина - работает для любого количества измерений"""
    n = len(x)
    return 10*n + np.sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

def rosenbrock_function(x):
    """Функция Розенброка - работает для любого количества измерений"""
    return np.sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 
                  for i in range(len(x)-1)])

def ackley_function(x):
    """Функция Акклея - работает для любого количества измерений"""
    n = len(x)
    sum1 = np.sum(np.array(x)**2)
    sum2 = np.sum(np.cos(2*np.pi*np.array(x)))
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e

# ============================================================================
# РЕАЛИЗАЦИЯ РАЗЛИЧНЫХ СТРАТЕГИЙ ЭВОЛЮЦИОННЫХ СТРАТЕГИЙ
# ============================================================================

class Individual:
    """Особь со стратегическими параметрами согласно уравнению (5.1)"""
    def __init__(self, x, sigma):
        self.x = np.array(x, dtype=np.float64)
        self.sigma = np.array(sigma, dtype=np.float64)
        self.fitness = None
    
    def __str__(self):
        return f"x: {self.x}, σ: {self.sigma}, fitness: {self.fitness}"

def one_plus_one_es(objective_func, bounds, max_generations=500, initial_sigma=0.3, 
                   success_rule_interval=50, verbose=True):
    """(1+1)-ES: Двукратная эволюционная стратегия"""
    dim = len(bounds)
    fitness_history = []
    sigma_history = []
    all_populations = []
    
    # Инициализация одного родителя
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    sigma = np.full(dim, initial_sigma)
    parent = Individual(x, sigma)
    parent.fitness = objective_func(parent.x)
    
    success_count = 0
    all_populations.append([parent.x.copy()])
    
    if verbose:
        print(f"(1+1)-ES: Начальная приспособленность = {parent.fitness:.6f}")
    
    for generation in range(max_generations):
        # Создание потомка мутацией
        offspring_x = parent.x + parent.sigma * np.random.normal(0, 1, dim)
        
        # Применение границ
        for i in range(dim):
            offspring_x[i] = np.clip(offspring_x[i], bounds[i][0], bounds[i][1])
        
        offspring_fitness = objective_func(offspring_x)
        
        # Отбор: замена если лучше
        if offspring_fitness < parent.fitness:
            parent.x = offspring_x.copy()
            parent.fitness = offspring_fitness
            success_count += 1
        
        fitness_history.append(parent.fitness)
        sigma_history.append(parent.sigma.copy())
        all_populations.append([parent.x.copy()])
        
        # Правило успеха 1/5
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
    """(μ,λ)-ES: Многократная эволюционная стратегия"""
    dim = len(bounds)
    population = []
    fitness_history = []
    all_populations = []
    
    # Инициализация популяции
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
        print(f"(μ,λ)-ES: Начальная лучшая приспособленность = {best_individual.fitness:.6f}")
    
    for generation in range(max_generations):
        offspring = []
        
        # Генерация λ потомков
        for _ in range(lambda_):
            # Выбор родителей турнирным отбором
            parents = []
            for _ in range(2):
                candidates = np.random.choice(population, min(3, len(population)), replace=False)
                best_candidate = min(candidates, key=lambda ind: ind.fitness)
                parents.append(best_candidate)
            
            # Рекомбинация
            x_child = np.mean([p.x for p in parents], axis=0)
            sigma_child = np.mean([p.sigma for p in parents], axis=0)
            
            # Мутация
            tau = 1.0 / np.sqrt(2 * np.sqrt(dim))
            sigma_child = sigma_child * np.exp(tau * np.random.normal(0, 1, dim))
            sigma_child = np.clip(sigma_child, 0.01, 2.0)
            
            x_child = x_child + sigma_child * np.random.normal(0, 1, dim)
            
            # Применение границ
            for i in range(dim):
                x_child[i] = np.clip(x_child[i], bounds[i][0], bounds[i][1])
            
            child = Individual(x_child, sigma_child)
            child.fitness = objective_func(child.x)
            offspring.append(child)
        
        # Отбор
        offspring.sort(key=lambda ind: ind.fitness)
        population = offspring[:mu]
        
        best_individual = population[0]
        fitness_history.append(best_individual.fitness)
        all_populations.append([ind.x.copy() for ind in population])
        
    return best_individual.x, best_individual.fitness, fitness_history, all_populations

def mu_plus_lambda_es(objective_func, bounds, mu=15, lambda_=45, max_generations=100,
                     initial_sigma=0.3, verbose=True):
    """(μ+λ)-ES: Многократная эволюционная стратегия"""
    dim = len(bounds)
    population = []
    fitness_history = []
    all_populations = []
    
    # Инициализация популяции
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
        print(f"(μ+λ)-ES: Начальная лучшая приспособленность = {best_individual.fitness:.6f}")
    
    for generation in range(max_generations):
        offspring = []
        
        # Генерация λ потомков
        for _ in range(lambda_):
            # Выбор родителей
            parents = []
            for _ in range(2):
                candidates = np.random.choice(population, min(3, len(population)), replace=False)
                best_candidate = min(candidates, key=lambda ind: ind.fitness)
                parents.append(best_candidate)
            
            # Рекомбинация
            x_child = np.mean([p.x for p in parents], axis=0)
            sigma_child = np.mean([p.sigma for p in parents], axis=0)
            
            # Мутация
            tau = 1.0 / np.sqrt(2 * np.sqrt(dim))
            sigma_child = sigma_child * np.exp(tau * np.random.normal(0, 1, dim))
            sigma_child = np.clip(sigma_child, 0.01, 2.0)
            
            x_child = x_child + sigma_child * np.random.normal(0, 1, dim)
            
            # Применение границ
            for i in range(dim):
                x_child[i] = np.clip(x_child[i], bounds[i][0], bounds[i][1])
            
            child = Individual(x_child, sigma_child)
            child.fitness = objective_func(child.x)
            offspring.append(child)
        
        # Отбор из родителей + потомков
        combined = population + offspring
        combined.sort(key=lambda ind: ind.fitness)
        population = combined[:mu]
        
        best_individual = population[0]
        fitness_history.append(best_individual.fitness)
        all_populations.append([ind.x.copy() for ind in population])
        
    return best_individual.x, best_individual.fitness, fitness_history, all_populations

# ============================================================================
# МОДИФИЦИРОВАННЫЕ ФУНКЦИИ ЭС ДЛЯ n-МЕРНОЙ ОПТИМИЗАЦИИ
# ============================================================================

def run_es_for_ndim(objective_func, bounds, strategy_name, n_dim=2, 
                   max_generations=100, verbose=True):
    """Запуск ЭС для n-мерной оптимизации с отслеживанием производительности"""
    print(f"\n--- Тестирование {strategy_name} для n={n_dim} ---")
    start_time = time.time()
    
    if strategy_name == '(1+1)-ES':
        best_x, best_fitness, fitness_history, sigma_history, pop_history = one_plus_one_es(
            objective_func, bounds, max_generations=max_generations, verbose=False
        )
    elif strategy_name == '(μ,λ)-ES':
        best_x, best_fitness, fitness_history, pop_history = mu_comma_lambda_es(
            objective_func, bounds, mu=15, lambda_=45, 
            max_generations=max_generations, verbose=False
        )
    else:  # (μ+λ)-ES
        best_x, best_fitness, fitness_history, pop_history = mu_plus_lambda_es(
            objective_func, bounds, mu=15, lambda_=45, 
            max_generations=max_generations, verbose=False
        )
    
    end_time = time.time()
    
    return {
        'best_solution': best_x,
        'best_fitness': best_fitness,
        'computation_time': end_time - start_time,
        'generations': len(fitness_history),
        'dimension': n_dim
    }

# ============================================================================
# ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# ============================================================================

def plot_function_with_trajectory(population_history, strategy_name, bounds):
    """Построение функции с траекторией оптимизации"""
    x1 = np.linspace(bounds[0][0], bounds[0][1], 100)
    x2 = np.linspace(bounds[1][0], bounds[1][1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = goldstein_price([X1[j, i], X2[j, i]])
    
    fig = plt.figure(figsize=(18, 6))
    
    # 3D поверхность с траекторией
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6)
    
    # Построение траектории оптимизации
    for gen, population in enumerate(population_history):
        if gen % max(1, len(population_history)//10) == 0:  # Выборка поколений
            color_val = gen / len(population_history)
            for individual in population:
                fitness = goldstein_price(individual)
                ax1.scatter(individual[0], individual[1], fitness, 
                           color=plt.cm.plasma(color_val), s=30, alpha=0.7)
    
    # Отметка важных точек
    if population_history:
        # Начальные точки
        for individual in population_history[0]:
            fitness = goldstein_price(individual)
            ax1.scatter(individual[0], individual[1], fitness, 
                       color='green', s=100, marker='o', label='Старт' if individual is population_history[0][0] else "")
        
        # Конечные точки  
        for individual in population_history[-1]:
            fitness = goldstein_price(individual)
            ax1.scatter(individual[0], individual[1], fitness, 
                       color='red', s=100, marker='*', label='Финиш' if individual is population_history[-1][0] else "")
    
    # Отметка глобального оптимума
    ax1.scatter(0, -1, 3, color='gold', s=200, marker='D', label='Глобальный оптимум', edgecolors='black')
    
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2)')
    ax1.set_title(f'{strategy_name}\n3D вид с траекторией оптимизации')
    ax1.legend()
    
    # Контурный график с траекторией
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X1, X2, Z, levels=20, alpha=0.6)
    plt.colorbar(contour, ax=ax2)
    
    # Построение траектории в 2D
    for gen, population in enumerate(population_history):
        if gen % max(1, len(population_history)//10) == 0:
            color_val = gen / len(population_history)
            x1_vals = [ind[0] for ind in population]
            x2_vals = [ind[1] for ind in population]
            ax2.scatter(x1_vals, x2_vals, color=plt.cm.plasma(color_val), 
                       alpha=0.6, s=20)
    
    # Отметка важных точек
    ax2.scatter(0, -1, color='gold', s=200, marker='D', label='Глобальный оптимум', edgecolors='black')
    
    if population_history:
        start_x1 = [ind[0] for ind in population_history[0]]
        start_x2 = [ind[1] for ind in population_history[0]]
        end_x1 = [ind[0] for ind in population_history[-1]]
        end_x2 = [ind[1] for ind in population_history[-1]]
        
        ax2.scatter(start_x1, start_x2, color='green', s=80, marker='o', label='Начальная популяция')
        ax2.scatter(end_x1, end_x2, color='red', s=80, marker='*', label='Финальная популяция')
    
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title(f'{strategy_name}\nКонтур с траекторией поиска')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График сходимости
    ax3 = fig.add_subplot(133)
    best_fitness_history = [min([goldstein_price(ind) for ind in pop]) for pop in population_history]
    ax3.plot(best_fitness_history, 'b-', linewidth=2, label='Лучшая приспособленность')
    ax3.axhline(y=3, color='r', linestyle='--', label='Теоретический минимум')
    ax3.set_xlabel('Поколение')
    ax3.set_ylabel('Приспособленность')
    ax3.set_title('История сходимости')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    if min(best_fitness_history) > 0:
        ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_initial_function():
    """Построение функции Гольдстейна-Прайса без траекторий"""
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            Z[j, i] = goldstein_price([X1[j, i], X2[j, i]])
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D поверхность
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    ax1.scatter(0, -1, 3, color='red', s=100, label='Глобальный оптимум')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2)')
    ax1.set_title('Функция Гольдстейна-Прайса (3D)')
    ax1.legend()
    
    # Контурный график
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(X1, X2, Z, levels=20)
    ax2.scatter(0, -1, color='red', s=100, label='Глобальный оптимум')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Контурный график')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(contour, ax=ax2)
    
    # Описание функции
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    ax3.text(0.1, 0.9, 'Функция Гольдстейна-Прайса', fontsize=14, fontweight='bold')
    ax3.text(0.1, 0.7, 'Глобальный минимум: f(0, -1) = 3.0', fontsize=12)
    ax3.text(0.1, 0.6, 'Пространство поиска: -2 ≤ x₁, x₂ ≤ 2', fontsize=12)
    ax3.text(0.1, 0.5, 'Характеристики:', fontsize=12, fontweight='bold')
    ax3.text(0.1, 0.4, '- Множество локальных минимумов', fontsize=10)
    ax3.text(0.1, 0.3, '- Сложный ландшафт', fontsize=10)
    ax3.text(0.1, 0.2, '- Сложна для оптимизации', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# КОМПЛЕКСНОЕ СРАВНЕНИЕ n=2 vs n=3
# ============================================================================

def compare_2d_vs_3d_performance():
    """Сравнение производительности ЭС между 2D и 3D оптимизацией"""
    print("=" * 70)
    print("КОМПЛЕКСНОЕ СРАВНЕНИЕ n=2 vs n=3")
    print("=" * 70)
    
    # Тестовые функции для сравнения
    test_functions = {
        'Сфера': sphere_function,
        'Растригин': rastrigin_function,
        'Розенброк': rosenbrock_function
    }
    
    strategies = ['(1+1)-ES', '(μ,λ)-ES', '(μ+λ)-ES']
    
    # Границы для разных функций
    bounds_2d = [(-5.12, 5.12), (-5.12, 5.12)]
    bounds_3d = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)]
    
    results_2d = {}
    results_3d = {}
    
    for func_name, objective_func in test_functions.items():
        print(f"\n{'='*50}")
        print(f"ТЕСТИРОВАНИЕ ФУНКЦИИ {func_name.upper()}")
        print(f"{'='*50}")
        
        results_2d[func_name] = {}
        results_3d[func_name] = {}
        
        # Тестирование n=2
        print(f"\n--- n=2 ИЗМЕРЕНИЯ ---")
        bounds_2d_actual = bounds_2d if func_name != 'Розенброк' else [(-2.048, 2.048), (-2.048, 2.048)]
        
        for strategy in strategies:
            result = run_es_for_ndim(objective_func, bounds_2d_actual, strategy, n_dim=2)
            results_2d[func_name][strategy] = result
            print(f"{strategy}: Приспособленность = {result['best_fitness']:.6f}, "
                  f"Время = {result['computation_time']:.3f}с")
        
        # Тестирование n=3  
        print(f"\n--- n=3 ИЗМЕРЕНИЯ ---")
        bounds_3d_actual = bounds_3d if func_name != 'Розенброк' else [(-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048)]
        
        for strategy in strategies:
            result = run_es_for_ndim(objective_func, bounds_3d_actual, strategy, n_dim=3)
            results_3d[func_name][strategy] = result
            print(f"{strategy}: Приспособленность = {result['best_fitness']:.6f}, "
                  f"Время = {result['computation_time']:.3f}с")
    
    return results_2d, results_3d

def plot_2d_vs_3d_comparison(results_2d, results_3d):
    """Построение комплексного сравнения между 2D и 3D производительностью"""
    functions = list(results_2d.keys())
    strategies = ['(1+1)-ES', '(μ,λ)-ES', '(μ+λ)-ES']
    
    # Создание сравнительных графиков
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Сравнение времени вычислений
    time_data_2d = []
    time_data_3d = []
    
    for func in functions:
        for strategy in strategies:
            time_data_2d.append(results_2d[func][strategy]['computation_time'])
            time_data_3d.append(results_3d[func][strategy]['computation_time'])
    
    x_pos = np.arange(len(functions) * len(strategies))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, time_data_2d, width, label='n=2', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, time_data_3d, width, label='n=3', alpha=0.7)
    
    axes[0, 0].set_xlabel('Функция × Стратегия')
    axes[0, 0].set_ylabel('Время вычислений (с)')
    axes[0, 0].set_title('Время вычислений: n=2 vs n=3')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Установка меток оси X
    labels = []
    for func in functions:
        for strategy in strategies:
            labels.append(f"{func}\n{strategy}")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
    
    # 2. Сравнение приспособленности (логарифмическая шкала)
    fitness_data_2d = []
    fitness_data_3d = []
    
    for func in functions:
        for strategy in strategies:
            fitness_data_2d.append(results_2d[func][strategy]['best_fitness'])
            fitness_data_3d.append(results_3d[func][strategy]['best_fitness'])
    
    x_pos = np.arange(len(functions) * len(strategies))
    
    axes[0, 1].bar(x_pos - width/2, fitness_data_2d, width, label='n=2', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, fitness_data_3d, width, label='n=3', alpha=0.7)
    
    axes[0, 1].set_xlabel('Функция × Стратегия')
    axes[0, 1].set_ylabel('Лучшая приспособленность')
    axes[0, 1].set_title('Качество решения: n=2 vs n=3')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right')
    
    # 3. Коэффициент ускорения (2D время / 3D время)
    speedup_factors = []
    speedup_labels = []
    
    for func in functions:
        for strategy in strategies:
            time_2d = results_2d[func][strategy]['computation_time']
            time_3d = results_3d[func][strategy]['computation_time']
            speedup = time_2d / time_3d if time_3d > 0 else 1
            speedup_factors.append(speedup)
            speedup_labels.append(f"{func}\n{strategy}")
    
    x_pos = np.arange(len(speedup_factors))
    colors = ['green' if x >= 1 else 'red' for x in speedup_factors]
    
    bars = axes[1, 0].bar(x_pos, speedup_factors, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Функция × Стратегия')
    axes[1, 0].set_ylabel('Коэффициент ускорения (2D/3D)')
    axes[1, 0].set_title('Вычислительное ускорение: 2D vs 3D\n(Зеленый = 2D быстрее, Красный = 3D быстрее)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(speedup_labels, rotation=45, ha='right')
    
    # Добавление значений на столбцы
    for bar, value in zip(bars, speedup_factors):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{value:.2f}x', ha='center', va='bottom', fontsize=8)
    
    # 4. Сравнение поколений
    gens_2d = []
    gens_3d = []
    
    for func in functions:
        for strategy in strategies:
            gens_2d.append(results_2d[func][strategy]['generations'])
            gens_3d.append(results_3d[func][strategy]['generations'])
    
    x_pos = np.arange(len(functions) * len(strategies))
    
    axes[1, 1].bar(x_pos - width/2, gens_2d, width, label='n=2', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, gens_3d, width, label='n=3', alpha=0.7)
    
    axes[1, 1].set_xlabel('Функция × Стратегия')
    axes[1, 1].set_ylabel('Поколения')
    axes[1, 1].set_title('Требуемые поколения: n=2 vs n=3')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def print_detailed_comparison_table(results_2d, results_3d):
    """Вывод детальной таблицы сравнения между 2D и 3D результатами"""
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ n=2 vs n=3")
    print("=" * 80)
    
    functions = list(results_2d.keys())
    strategies = ['(1+1)-ES', '(μ,λ)-ES', '(μ+λ)-ES']
    
    for func in functions:
        print(f"\nФУНКЦИЯ {func.upper()}:")
        print("-" * 70)
        print(f"{'Стратегия':<12} {'Измерение':<10} {'Приспособл.':<12} {'Время (с)':<10} {'Поколения':<12} {'Ускорение'}")
        print("-" * 70)
        
        for strategy in strategies:
            res_2d = results_2d[func][strategy]
            res_3d = results_3d[func][strategy]
            
            speedup = res_2d['computation_time'] / res_3d['computation_time']
            speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x (медленнее)"
            
            print(f"{strategy:<12} {'n=2':<10} {res_2d['best_fitness']:<12.6f} "
                  f"{res_2d['computation_time']:<10.3f} {res_2d['generations']:<12} {speedup_str}")
            
            print(f"{'':<12} {'n=3':<10} {res_3d['best_fitness']:<12.6f} "
                  f"{res_3d['computation_time']:<10.3f} {res_3d['generations']:<12}")

def compare_es_strategies(objective_func, bounds):
    """Сравнение различных стратегий ЭС"""
    print("=" * 70)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ЭВОЛЮЦИОННЫХ СТРАТЕГИЙ")
    print("=" * 70)
    
    strategies = {
        '(1+1)-ES': one_plus_one_es,
        '(μ,λ)-ES': mu_comma_lambda_es,
        '(μ+λ)-ES': mu_plus_lambda_es
    }
    
    results = {}
    
    for name, strategy_func in strategies.items():
        print(f"\n--- Запуск {name} ---")
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
            
            print(f"Лучшая приспособленность: {best_fitness:.8f}")
            print(f"Лучшее решение: x1={best_x[0]:.6f}, x2={best_x[1]:.6f}")
            print(f"Время вычислений: {end_time - start_time:.4f}с")
            print(f"Поколений: {len(fitness_history)}")
            print(f"Ошибка от оптимума: {results[name]['error']:.8f}")
            
            # Построение траектории для этой стратегии
            print(f"Построение траектории для {name}...")
            plot_function_with_trajectory(pop_history, name, bounds)
            
        except Exception as e:
            print(f"Ошибка в {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def parameter_sensitivity_analysis(objective_func, bounds):
    """Анализ чувствительности параметров"""
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ")
    print("=" * 70)
    
    # Тестирование различных размеров популяции
    print("\n1. Чувствительность к размеру популяции (для (μ,λ)-ES):")
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
        
        print(f"Популяция {pop_size}: Приспособленность = {best_fitness:.6f}, "
              f"Время = {end_time - start_time:.3f}с")
    
    # Тестирование различных параметров мутации
    print("\n2. Чувствительность к начальной силе мутации (для (1+1)-ES):")
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
        
        print(f"Сигма {sigma}: Приспособленность = {best_fitness:.6f}, "
              f"Время = {end_time - start_time:.3f}с")
    
    return pop_results, sigma_results

# ============================================================================
# ОСНОВНОЕ ИСПОЛНЕНИЕ
# ============================================================================

if __name__ == "__main__":
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ЭВОЛЮЦИОННЫХ СТРАТЕГИЙ")
    print("Оптимизация функции Гольдстейна-Прайса")
    print("=" * 70)
    
    # Известный глобальный минимум
    true_optimum = np.array([0, -1])
    true_minimum = 3.0
    
    print(f"Теоретический глобальный минимум: f({true_optimum[0]}, {true_optimum[1]}) = {true_minimum}")
    
    # Определение границ
    bounds = [(-2, 2), (-2, 2)]
    
    # 1. Построение начальной функции
    print("\n1. ПОСТРОЕНИЕ ФУНКЦИИ ГОЛЬДСТЕЙНА-ПРАЙСА...")
    plot_initial_function()
    
    # 2. Сравнение всех стратегий ЭС с визуализацией траекторий
    print("\n2. СРАВНЕНИЕ СТРАТЕГИЙ ЭС С ВИЗУАЛИЗАЦИЕЙ ТРАЕКТОРИЙ...")
    results = compare_es_strategies(goldstein_price, bounds)
    
    # 3. Анализ чувствительности параметров
    print("\n3. ВЫПОЛНЕНИЕ АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ...")
    pop_results, sigma_results = parameter_sensitivity_analysis(goldstein_price, bounds)
    
    # 4. n=2 vs n=3 КОМПЛЕКСНОЕ СРАВНЕНИЕ (НОВОЕ ТРЕБОВАНИЕ)
    print("\n4. ВЫПОЛНЕНИЕ n=2 vs n=3 СРАВНЕНИЯ...")
    results_2d, results_3d = compare_2d_vs_3d_performance()
    
    # 5. Построение результатов сравнения
    print("\n5. ПОСТРОЕНИЕ РЕЗУЛЬТАТОВ СРАВНЕНИЯ n=2 vs n=3...")
    plot_2d_vs_3d_comparison(results_2d, results_3d)
    
    # 6. Вывод детальной таблицы сравнения
    print_detailed_comparison_table(results_2d, results_3d)
    
    # 7. Отображение финальных результатов для Гольдстейна-Прайса
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА РЕЗУЛЬТАТОВ - ФУНКЦИЯ ГОЛЬДСТЕЙНА-ПРАЙСА")
    print("=" * 70)
    
    if results:
        best_strategy = min(results.items(), key=lambda x: x[1]['best_fitness'])
        print(f"Лучшая стратегия: {best_strategy[0]}")
        print(f"Лучшая приспособленность: {best_strategy[1]['best_fitness']:.8f}")
        print(f"Решение: x1 = {best_strategy[1]['best_solution'][0]:.8f}, "
              f"x2 = {best_strategy[1]['best_solution'][1]:.8f}")
        print(f"Ошибка от теоретического оптимума: {best_strategy[1]['error']:.8f}")
        print(f"Время вычислений: {best_strategy[1]['computation_time']:.4f}с")
        
        # Отображение всех результатов в таблице
        print("\n" + "-" * 70)
        print("ПРОИЗВОДИТЕЛЬНОСТЬ ВСЕХ СТРАТЕГИЙ:")
        print("-" * 70)
        for name, data in results.items():
            print(f"{name}: Приспособленность = {data['best_fitness']:.6f}, "
                  f"Время = {data['computation_time']:.3f}с, "
                  f"Ошибка = {data['error']:.6f}")
    
    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 70)