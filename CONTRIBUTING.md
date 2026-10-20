# Contributing to ChurnSight

Thank you for your interest in contributing to **ChurnSight**!

---

## Development Workflow

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/Anjelo-Sherin/day-03-customer-churn-predictor.git
   cd day-03-customer-churn-predictor
   ```

2. **Set Up Development Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Code Style & ML Standards:**
   - Adhere to strict featurization ordering: **never** fit transformers across the full dataset prior to splitting.
   - Preserve all missingness indicators and explicit validation routines.
   - Follow PEP 8 guidelines and include type hints across all function signatures.

4. **Run Tests Prior to Submitting PR:**
   ```bash
   pytest -v
   ```
   Ensure all 6 tests pass without regressions.

5. **Submit a Pull Request:**
   - Write clear, conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`).
   - Describe the business rationale and provide benchmark comparison metrics when modifying modeling components.
