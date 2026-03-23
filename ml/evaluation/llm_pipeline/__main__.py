"""Allows running the package directly: python -m evaluation.llm_pipeline"""

import sys
from .cli import main

sys.exit(main())
