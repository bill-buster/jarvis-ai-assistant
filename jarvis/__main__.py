"""Entry point for python -m jarvis execution.

This module allows running JARVIS as a module:
    python -m jarvis chat
    python -m jarvis health
    python -m jarvis setup
    python -m jarvis --help
"""

from jarvis.cli import run

if __name__ == "__main__":
    run()
