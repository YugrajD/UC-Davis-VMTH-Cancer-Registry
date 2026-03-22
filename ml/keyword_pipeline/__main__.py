"""Allows running the package directly: python -m keyword_pipeline"""

from .cli import main
import sys

sys.exit(main())
