# mlops/optuna_tuner.py
# Optuna hyperparameter search for EfficientNet-B0 fine-tuning
# 10 trials: lr, dropout, batch_size
# All trials logged to MLflow on DagsHub
# Run on Kaggle T4 — not locally

import os
import optuna
import mlflow
import dagshub
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


MVTEC_PATH = os.environ.get("MVTEC_PATH", "/kaggle/input/datasets/ipythonx/mvtec-ad")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS   = 10


class MVTecBinaryDataset(Dataset):
    """
    Binary classification dataset: normal=0, defective=1.
    Used only for EfficientNet fine-tuning (GradCAM++ quality).
    NOT used for PatchCore training.
    """

    def __init__(self, mvtec_path: str, transform=None):
        self.samples   = []
        self.transform = transform
        categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
            'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
            'transistor', 'wood', 'zipper'
        ]

        for cat in categories:
            # Normal
            train_dir = os.path.join(mvtec_path, cat, "train", "good")
            for f in os.listdir(train_dir):
                if f.endswith((".png", ".jpg")):
                    self.samples.append(
                        (os.path.join(train_dir, f), 0)
                    )
            # Defective
            test_dir = os.path.join(mvtec_path, cat, "test")
            for defect_type in os.listdir(test_dir):
                if defect_type == "good":
                    continue
                d_dir = os.path.join(test_dir, defect_type)
                for f in os.listdir(d_dir):
                    if f.endswith((".png", ".jpg")):
                        self.samples.append(
                            (os.path.join(d_dir, f), 1)
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_model(dropout: float) -> nn.Module:
    model = models.efficientnet_b0(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(1280, 2)
    )
    return model.to(DEVICE)


def train_one_trial(trial):
    """Single Optuna trial — returns validation AUC."""
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout    = trial.suggest_float("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset    = MVTecBinaryDataset(MVTEC_PATH, transform=transform)
    n_val      = int(0.2 * len(dataset))
    n_train    = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                               shuffle=False, num_workers=2)

    model     = build_model(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train 3 epochs per trial
    for epoch in range(3):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

    # Validate
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)[:, 1]
            all_scores.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_scores)

    # Log trial to MLflow
    with mlflow.start_run(run_name=f"efficientnet_trial_{trial.number}",
                           nested=True):
        mlflow.log_param("lr",         lr)
        mlflow.log_param("dropout",    dropout)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_metric("val_auc",   auc)

    return auc


def run_optuna_search():
    dagshub.init(repo_owner="devangmishra1424",
                 repo_name="AnomalyOS", mlflow=True)

    with mlflow.start_run(run_name="efficientnet_optuna_search"):
        study = optuna.create_study(direction="maximize")
        study.optimize(train_one_trial, n_trials=N_TRIALS)

        best = study.best_trial
        print(f"\nBest trial: AUC={best.value:.4f}")
        print(f"  lr={best.params['lr']:.6f}")
        print(f"  dropout={best.params['dropout']:.3f}")
        print(f"  batch_size={best.params['batch_size']}")

        mlflow.log_metric("best_val_auc", best.value)
        mlflow.log_params(best.params)

    return best.params


if __name__ == "__main__":
    run_optuna_search()