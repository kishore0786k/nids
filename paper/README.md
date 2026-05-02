# Neuro-Symbolic NIDS Paper Package

This folder contains generated IEEE-style research artifacts for the NF-ToN-IoT-V2 neuro-symbolic intrusion detection project.

## Generate Artifacts

Run from the project root:

```powershell
.\build_publication_package.bat
```

or:

```powershell
cd backend
..\venv\Scripts\python.exe generate_publication_package.py
```

## Outputs

- `paper/generated/*.tex`: abstract text, setup table, comparison table, algorithm outline, and results paragraph.
- `paper/figures/generated/*.png`: high-resolution figure previews.
- `paper/figures/generated/*.pdf`: publication figure files.
- `results/publication_package/publication_package.json`: reproducibility artifact.
- `results/publication_package/publication_summary.md`: compact results summary.
- `results/publication_package/submission_readiness.md`: remaining manual checks.

## Submission Notes

This package gives the project a research-software structure. Before IEEE submission, verify literature baselines, references, train/test split details, and claims in the manuscript.
