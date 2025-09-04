import logging
from logging.handlers import RotatingFileHandler
import time
import os

logging.getLogger("faker").setLevel(logging.WARNING)

def setup_logging(log_path="logs.log", file_level=logging.DEBUG, enable_console=False):


    # Ensure directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # master level
    root_logger.addHandler(file_handler)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def log_time(logger=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            active_logger = logger or logging.getLogger(func.__module__)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            active_logger.info("%s executed in %.3f seconds", func.__name__, duration)
            return result
        return wrapper
    return decorator

def suppress_module_output(module_names, suppress_console=True, suppress_file=False):
    if isinstance(module_names, str):
        module_names = [module_names]

    class SuppressFilter(logging.Filter):
        def filter(self, record):
            return not any(record.name.startswith(name) for name in module_names)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if suppress_console and isinstance(handler, logging.StreamHandler):
            handler.addFilter(SuppressFilter())
        if suppress_file and isinstance(handler, logging.Handler) and not isinstance(handler, logging.StreamHandler):
            handler.addFilter(SuppressFilter())