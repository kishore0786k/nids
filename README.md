# Neuro-Symbolic NIDS Research Package

Publication-oriented intrusion detection project for NF-ToN-IoT-V2 NetFlow traffic.

## Run Dashboard

```bat
run_project.bat
```

Then open:

```text
http://127.0.0.1:5000
```

## Build IEEE/Overleaf Artifacts

```bat
build_publication_package.bat
```

Generated outputs:

- `paper/generated/*.tex`
- `paper/figures/generated/*.pdf`
- `results/publication_package/publication_package.json`
- `NIDS_IEEE_Overleaf_Package.zip`

## Structure

- `backend/app.py`: Flask API and dashboard server.
- `backend/nids_engine.py`: model, metrics, charts, reliability, OOD, and defence backend.
- `backend/neuro_symbolic.py`: auditable symbolic rules.
- `frontend/`: dashboard UI.
- `paper/`: IEEE support files and generated paper assets.
- `results/publication_package/`: reproducibility package.

## Verification

```bat
cd backend
..\venv\Scripts\python.exe -m unittest test_smoke.py
```

## Final Submission Notes

Before IEEE submission, manually verify the cited baseline method, bibliography metadata, author details, and manuscript claims against generated artifacts.
