# src/marketing_simulation/logging_utils.py
import logging
from pathlib import Path
from datetime import datetime

_CONFIGURED = False

def init_logging(*, reset: bool = False, level: int = logging.INFO) -> None:
    """
    Configure the ROOT logger once for the whole app.
    - Writes to stdout AND logs/logs.txt
    - If reset=True, overwrite logs/logs.txt on this run.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "logs.txt"

    root = logging.getLogger()  # root logger
    root.setLevel(level)

    # Clear any pre-existing handlers (e.g., from notebooks/tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (overwrite if reset)
    mode = "w" if reset else "a"
    fh = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Optional header line so each run is obvious
    root.info("----- RUN START %s -----", datetime.utcnow().isoformat() + "Z")

    _CONFIGURED = True

def get_logger(name: str | None = None) -> logging.Logger:
    """Grab a namespaced logger. Use __name__ from each module."""
    return logging.getLogger(name or "app")
