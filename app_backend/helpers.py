import logging


# Setup logging

logging.basicConfig(
    filename="project_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Safe execution decorator

def safe_execute(interactive=False):
    """
    Decorator for safe execution of functions/methods.

    Parameters:
        interactive (bool): 
            If True, it will ask the user again for input if ValueError or TypeError occurs.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            while True:  # Loop only if interactive=True
                try:
                    return func(*args, **kwargs)  # run the original function
                except FileNotFoundError as e:
                    print(f"File not found: {e}")
                    break  # Cannot retry for file errors
                except KeyError as e:
                    print(f"Missing column/key: {e}")
                    break
                except ValueError as e:
                    print(f"Invalid value: {e}")
                    if interactive:
                        # Ask user for input again
                        if 'input_value' in kwargs:
                            kwargs['input_value'] = input("Please enter a valid value: ")
                        continue
                    break
                except TypeError as e:
                    print(f"Type error: {e}")
                    if interactive:
                        continue
                    break
                except ZeroDivisionError:
                    print("Cannot divide by zero!")
                    break
                except MemoryError:
                    print("Memory Error: Data too large!")
                    break
                except Exception as e:
                    logging.error(f"Unexpected error in {func.__name__}", exc_info=True)
                    print(f"An unexpected error occurred: {e}")
                    break
                finally:
                    pass  # optional cleanup code
        return wrapper
    return decorator
