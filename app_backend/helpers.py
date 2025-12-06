import logging
from functools import wraps

# Setup logging
logging.basicConfig(
    filename="project_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def safe_execute(interactive=False):
    """
    Decorator for safe execution of methods (not __init__).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            while True:  # Only loops if interactive=True
                try:
                    return func(*args, **kwargs)
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    break
                except KeyError as e:
                    print(f"Missing column/key: {e}")
                    break
                except ValueError as e:
                    print(f"Invalid value: {e}")
                    if interactive:
                        continue
                    break
                except TypeError as e:
                    print(f"Type error: {e}")
                    if interactive:
                        continue
                    break
                except Exception as e:
                    logging.error(f"Unexpected error in {func.__name__}", exc_info=True)
                    print(f"An unexpected error occurred: {e}")
                    break
        return wrapper
    return decorator
