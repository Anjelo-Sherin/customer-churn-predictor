# Installation & Setup Guide — ChurnSight

This guide walks you through setting up ChurnSight locally or running it with Docker.

---

## Prerequisites

- **Python:** 3.9, 3.10, 3.11, or 3.12
- **Operating System:** macOS (Apple Silicon or Intel), Linux (Ubuntu, Debian, Fedora), or Windows (WSL2 recommended)
- **Git**

---

## Option 1: Local Virtual Environment Setup (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/Anjelo-Sherin/day-03-customer-churn-predictor.git
cd day-03-customer-churn-predictor
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Training Pipeline (Pre-computes artifacts & models)
```bash
python scripts/generate_notebook_and_charts.py
```
*Note: Trained artifacts (`models/churn_artifacts.joblib`) are already bundled with the repository for instant zero-setup launch.*

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## Option 2: Docker Setup (Zero Configuration)

Build and run ChurnSight in a standardized container with all system dependencies and OpenMP runtimes pre-configured:

```bash
# Build the Docker image
docker build -t churnsight:latest .

# Run the container mapping port 8501
docker run -p 8501:8501 churnsight:latest
```
Access the dashboard at **`http://localhost:8501`**.

---

## Running Automated Tests

Run the full automated test suite (including pipeline validation, data cleaning checks, and Streamlit `AppTest` headless UI tests):

```bash
pytest -v
```

All tests should pass:
```
tests/test_app_ui.py::test_streamlit_app_loads_and_renders_cleanly PASSED
tests/test_app_ui.py::test_streamlit_sidebar_and_threshold_slider PASSED
tests/test_app_ui.py::test_streamlit_dataframes_and_charts_present PASSED
tests/test_churn_pipeline.py::test_data_quality_and_whitespace_total_charges PASSED
tests/test_churn_pipeline.py::test_strict_split_and_preprocessor_no_leakage PASSED
tests/test_churn_pipeline.py::test_end_to_end_training_shap_and_what_if PASSED
```
