import time
from functools import wraps


def log_execution(func):
    '''
    Декоратор для получения длительности выполнения функции.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Функция {func.__name__} начала выполнение.")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Функция {func.__name__} закончила выполнение. Время выполнения: {elapsed_time:.4f} секунд.")
        return result
    return wrapper