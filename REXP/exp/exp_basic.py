import os
from typing import Any, Callable
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path

from wandb.wandb_run import Run
from ..models.baseModule import BaseModule
from ..utils.tools import EarlyStop

_EarlyStopType = EarlyStop

class EXP_BASIC():
    early_stop: _EarlyStopType|None = None
    result_path: str | Path
    epochs: int
    wandb_logger: Run
    model: BaseModule | DDP

    def __init__(self, model, logger: Run, early_stop: _EarlyStopType|None = None, epochs=500, exp_id=os.environ['EXP_ID'],
                 result_path: Path|str=os.environ['RES_PATH'], device=os.environ['ACCELERATOR'], model_path: Path|str = "") -> None:
        self.logger = logging.getLogger(exp_id)
        self.logger.info("Initializing experiment...")
        self.logger.info("Checking device...")
        device_id = os.environ.get("LOCAL_RANK","0")
        self.device = torch.device(f"cuda:{device_id}") if device != 'cpu' else torch.device("cpu")
        os.makedirs(result_path, exist_ok=True)
        self.result_path = Path(result_path)
        # model = self._load_model()
        self.logger.info(f"Model loaded to device:{self.device}")
        self.model = DDP(model.to(self.device), device_ids=[self.device], find_unused_parameters=True) if device == 'torchrun' else model.to(self.device)
        
        # hyper parameters
        self.early_stop = early_stop
        self.epochs = epochs
        self.model_path = model_path
         
        # wandb logger
        self.wandb_logger = logger
        self.step = 0

    # def save_configure(self):
    #     with open(os.path.join(self.result_path, "config.json"), "w+") as f:
    #         json.dump(self.settings, f, indent=4)
    #     self.logger.info("Configuration saved.")

    def _get_data(self):
        raise NotImplementedError

    def _load_model(self) -> BaseModule:
        raise NotImplementedError("Please implement the _load_model method in your experiment class.")
    
    def train(self):
        pass

    def vali(self) -> np.floating[Any]:
        return np.floating(0.0)

    def test(self, test_loader):
        pass