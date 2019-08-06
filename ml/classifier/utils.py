from pathlib import Path
from typing import List

import numpy as np
import torch


def subdirs(p: Path) -> List[Path]:
    return [x for x in p.iterdir() if x.is_dir()]


class EarlyStopping:
    def __init__(self, patience, model_save_path, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf
        self.best_acc = 0
        self.model_save_path = model_save_path

    def __call__(self, val_loss, val_acc, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model):
        torch.save(model.state_dict(), self.model_save_path)
        self.best_loss = val_loss
        self.best_acc = val_acc
