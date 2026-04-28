"""Allows running the package directly: python -m annotation.llm"""

import sys
from .cli import main

sys.exit(main())
