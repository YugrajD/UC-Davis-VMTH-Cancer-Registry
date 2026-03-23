"""Allows running the package directly: python -m annotation --method keyword|llm"""

import sys
from .cli import main

sys.exit(main())
