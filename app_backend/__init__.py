# app_backend/__init__.py

"""
App Backend Package
-------------------
This package contains all backend modules for the DA_project:
- Data loading
- Data cleaning
- EDA analysis
- Visualization
- Helpers / safe execution
"""

# Core modules
from .data_cleaning import DataCleaning
from .data_loader import DataLoader
from .eda_analysis import EDAEngine
from .viz_charts import VizEngine
from .helpers import safe_execute


# Optional: define what is exposed on import *
__all__ = [
    "DataCleaning",   # Main data cleaning class
    "data_loader",    # Data loader class
    "df_copy",        # Helper to copy data
    "EDAEngine",      # EDA analysis engine
    "safe_execute",   # Decorator for safe execution
    "VizEngine"       # Visualization engine
]

