## Contributing

Thanks for your interest in contributing.

### Scope

This repository contains research code supporting the paper **"Regime Labels Are Not Representation-Invariant"**.
Contributions that improve **reproducibility**, **clarity**, and **robustness** are welcome.

### Development setup

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Suggested workflow

- Create a feature branch
- Keep changes small and focused
- Prefer readable code over micro-optimizations

### Style

- Keep imports at the top of files and grouped (stdlib / third-party / local)
- Add short English docstrings for public functions and non-obvious logic
- Avoid committing large generated artifacts under `outputs/` unless required for a release

### Reporting issues

Please include:
- Your OS and Python version
- The command you ran (e.g. `python run.py`)
- Relevant logs from `outputs/<asset>/results/run.log` (if available)
