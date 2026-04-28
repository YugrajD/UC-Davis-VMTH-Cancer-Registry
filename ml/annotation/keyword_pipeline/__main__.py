"""Allows running the package directly: python -m annotation.keyword"""

from .cli import main
import sys

sys.exit(main())
