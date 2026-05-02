"""Deprecated compatibility module.

Path handling is centralized in `src.project_paths`; adversarial robust-model
training lives in `src.adversarial_training`.
"""

from src.adversarial_training import FGSMAdversarialTrainer


if __name__ == "__main__":
    print("Use: python -m src.adversarial_training")
