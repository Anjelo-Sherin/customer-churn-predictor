"""
ChurnSight — Day 03: Customer Churn Predictor with Explainable AI Dashboard.

Core package providing:
- Leak-free data cleaning, schema validation, and feature engineering (`data_pipeline`)
- Multi-model Stratified CV, cost-sensitive threshold tuning, and slice auditing (`model_trainer`)
- TreeSHAP / LinearSHAP local and global explanations in plain business words (`explainer`)
- Counterfactual What-If simulation and ROI-ranked retention action engine (`action_engine`)
"""

import ctypes
import sys
from pathlib import Path

__version__ = "1.0.0"


def _ensure_macos_openmp() -> None:
    """
    On macOS arm64 without Homebrew `libomp`, preloads scikit-learn's bundled
    `libomp.dylib` so `xgboost` loads seamlessly in any virtualenv.
    """
    if sys.platform != "darwin":
        return
    try:
        import sklearn
        omp_path = Path(sklearn.__file__).resolve().parent / ".dylibs" / "libomp.dylib"
        if omp_path.exists():
            ctypes.CDLL(str(omp_path), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


_ensure_macos_openmp()
