"""Allows running the package directly: python -m petbert_pipeline"""

from .cli import main
import sys

sys.exit(main())
