# PythonProject8

## Requirements

- **Python 3.10** is required. Please ensure you are using Python 3.10 (not 3.11 or later).

## Installation

1. **Clone the repository** (if not already done):

   ```sh
   git clone <repo-url>
   cd PythonProject8
   ```

2. **(Recommended) Create and activate a virtual environment:**

   ```sh
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   - If you encounter issues with `opencv-python` or `mediapipe`, ensure you are using Python 3.10.

## Dependencies

- numpy
- pandas
- opencv-python
- mediapipe
- joblib
- matplotlib
- seaborn
- scikit-learn

## Data

- Data files are located in the `data/` directory.

## Notebooks

- See `EDA.ipynb` and `model.ipynb` for exploratory data analysis and modeling steps.

---

**Note:** All required dependencies are listed in `requirements.txt`. If you add new dependencies, update both this README and `requirements.txt` using:

```sh
pip freeze > requirements.txt
```
