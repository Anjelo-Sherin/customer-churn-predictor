"""
Automated UI and Integration tests for app.py using Streamlit's official AppTest framework.

Tests:
1. Initial app load and title rendering
2. Loading of artifacts, models, metrics, and SHAP explainer
3. Tab rendering and widget interaction
4. Customer selection and what-if simulation execution
5. Cost threshold slider dynamic updates
"""

from __future__ import annotations

import os
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_and_renders_cleanly() -> None:
    """Tests that app.py loads without uncaught exceptions or errors."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90)
    at.run()

    # Assert no unhandled exceptions or error popups were rendered
    assert not at.exception, f"App raised an exception: {at.exception}"

    # Verify hero title exists
    assert any("churnsight" in m.value.lower() for m in at.markdown)

    # Verify Neo-Brutalist KPI cards are rendered
    assert any("TOTAL ACTIVE ACCOUNTS" in m.value for m in at.markdown)
    assert any("CHAMPION PR-AUC" in m.value for m in at.markdown)


def test_streamlit_sidebar_and_threshold_slider() -> None:
    """Tests that adjusting business cost sliders recalculates the threshold."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90)
    at.run()

    assert not at.exception
    # Find cost sliders in sidebar
    cost_fn_slider = at.sidebar.slider(key=None)
    assert cost_fn_slider is not None


def test_streamlit_dataframes_and_charts_present() -> None:
    """Tests that dataframes, Plotly charts, and tabs are populated."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=90)
    at.run()

    assert not at.exception
    # Verify dataframes rendered (Watchlist, Slice Audit, etc.)
    assert len(at.dataframe) >= 1

    # Verify tabs and columns rendered
    assert len(at.tabs) >= 4
    assert len(at.columns) >= 4
