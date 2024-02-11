# type: ignore[attr-defined]
"""Neural Network from ground up."""

from importlib import metadata

from .graph.ops import *
from .graph.trace import Node, trace

__version__ = metadata.version(__name__)
