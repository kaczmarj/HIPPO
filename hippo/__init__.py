from hippo.models.abmil import AttentionMILModel
from hippo.search import greedy_search, maximize, minimize, smallest_difference

try:
    from hippo._version import __version__
except ImportError:
    __version__ = "0+unknown"

__all__ = ["AttentionMILModel", "greedy_search", "maximize", "minimize", "smallest_difference"]
