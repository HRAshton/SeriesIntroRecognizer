# Installation

## Requirements

- Python ≥ 3.12  
- Dependencies: `librosa`, `numpy`, `scipy`, `scikit-learn`, `cupy`, `pytest`

---

## Setup

```bash
git clone https://github.com/HRAshton/SeriesIntroRecognizer.git
cd SeriesIntroRecognizer
pip install -r requirements.txt
```

For GPU support, install CuPy manually for your CUDA version:

```bash
pip install cupy-cuda12x
```

---

## Test Verification

Run all tests:

```bash
pytest
```
