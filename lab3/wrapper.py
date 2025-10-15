# wrapper.py
import builtins
import runpy
import sys
import io
from typing import Any

# --- Словарь переводов (ключи — английские фрагменты, значения — русские) ---
TRANSLATIONS = {
    # Заголовки/общие
    "GENETIC ALGORITHM - KNAPSACK PROBLEM ANALYSIS": "🧬 ГЕНЕТИЧЕСКИЙ АЛГОРИТМ - АНАЛИЗ ЗАДАЧИ KNAPSACK",
    "TASK 1: SIMPLE PROBLEM (P07) - OPTIMAL RUN": "ЗАДАНИЕ 1: ПРОСТАЯ ЗАДАЧА (P07) - ОПТИМАЛЬНЫЙ ЗАПУСК",
    "TASK 2: COMPLEX PROBLEM (Set 7)": "ЗАДАНИЕ 2: СЛОЖНАЯ ЗАДАЧА (Set 7)",
    "PARAMETER SENSITIVITY ANALYSIS": "АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ПАРАМЕТРОВ",
    "ENCODING TYPE COMPARISON": "СРАВНЕНИЕ ТИПОВ КОДИРОВАНИЯ",
    "FINAL SUMMARY WITH OPTIMAL COMPARISON": "ФИНАЛЬНОЕ РЕЗЮМЕ С СРАВНЕНИЕМ ОПТИМУМА",
    "GENERATION": "Поколение",
    "Generation": "Поколение",
    # Мелкие фрагменты внутри строк
    "Best =": "Лучший =",
    "Avg =": "Средний =",
    "Diversity =": "Разнообразие =",
    "Best Fitness": "Лучший fitness",
    "Average Fitness": "Средний fitness",
    "Worst Fitness": "Худший fitness",
    "Fitness Convergence": "Сходимость fitness",
    "Population Diversity Over Time": "Разнообразие популяции во времени",
    "Final Population Fitness Distribution": "Распределение fitness финальной популяции",
    "Fitness Improvement Per Generation": "Улучшение fitness в поколение",
    "Performance vs Parameters": "Производительность vs Параметры",
    "Solution Comparison": "Сравнение решений",
    "SOLUTION DETAILS:": "ДЕТАЛИ РЕШЕНИЯ:",
    "Items selected:": "Выбрано предметов:",
    "Total weight:": "Общий вес:",
    "Solution vector:": "Вектор решения:",
    "Best Fitness Found:": "Лучший найденный fitness:",
    "Optimal Fitness:": "Оптимальный fitness:",
    "Accuracy:": "Точность:",
    "Status:": "Статус:",
    "Best Parameter Setting:": "Лучшая настройка параметров:",
    "Fitness with Best Parameters:": "Fitness при лучших параметрах:",
    "Best Encoding Type:": "Лучший тип кодирования:",
    "Fitness with Best Encoding:": "Fitness при лучшем кодировании:",
    "Population Size:": "Размер популяции:",
    "Crossover Rate:": "Вероятность кроссовера:",
    "Mutation Rate:": "Вероятность мутации:",
    "Encoding:": "Кодирование:",
    "OPTIMAL FOUND!": "ОПТИМАЛЬНОЕ РЕШЕНИЕ НАЙДЕНО!",
    "Very Close": "Очень близко",
    "TRUE OPTIMAL FOUND": "ИСТИННЫЙ ОПТИМУМ НАЙДЕН",
    "Closest to optimal": "Ближайший к оптимуму",
    # даты/метки
    "Execution Time:": "Время выполнения:",
    "EXECUTION TIME:": "ВРЕМЯ ВЫПОЛНЕНИЯ:",
    "CONVERGED AT GENERATION:": "СХОДИТСЯ НА ПОКОЛЕНИИ:",
    "FINAL DIVERSITY:": "ФИНАЛЬНОЕ РАЗНООБРАЗИЕ:",
    # добавьте сюда другие строковые фрагменты, которые хотите перевести
}

# --- Функция перевода строки ---
def translate_text(s: str) -> str:
    """Заменяет в строке все найденные фрагменты по TRANSLATIONS."""
    # Выполняем замену длинных ключей первыми (чтобы избежать перекрытий)
    # Отсортируем ключи по длине убыв.
    for k in sorted(TRANSLATIONS.keys(), key=len, reverse=True):
        if k in s:
            s = s.replace(k, TRANSLATIONS[k])
    return s

# --- Замена builtins.print ---
_original_print = builtins.print

def translated_print(*args: Any, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
    # Собираем строковый результат как оригинальный print сделал бы
    stream = io.StringIO()
    _original_print(*args, sep=sep, end="", file=stream, flush=flush)
    text = stream.getvalue()
    # Переводим текст
    try:
        text_translated = translate_text(text)
    except Exception:
        text_translated = text  # на случай ошибок — вернуть оригинал
    # Печатаем уже переведённый текст настоящим print
    _original_print(text_translated, end=end, file=file, flush=flush)

# Патчим print
builtins.print = translated_print

# --- Запуск целевого скрипта переданого в аргументе командной строки ---
def main():
    if len(sys.argv) < 2:
        _original_print("Usage: python wrapper.py your_script.py", file=sys.stderr)
        sys.exit(1)
    target = sys.argv[1]
    # Передать дополнительные аргументы скрипту, если есть
    sys.argv = sys.argv[1:]
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:
        # Пропускаем SystemExit, чтобы обёртка не падала
        pass
    except Exception as e:
        # В случае исключения — печатаем трассировку (она тоже будет переведена по словарю, где возможно)
        import traceback
        _original_print("Error while running target script:", file=sys.stderr)
        traceback.print_exc()

if __name__ == "__main__":
    main()
