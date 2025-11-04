import os
from typing import Any
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from pathlib import Path

import wandb
from wandb.wandb_run import Run
from ..models.baseModule import BaseModule
from ..utils.tools import EarlyStop, setup_logger, get_result_path

_EarlyStopType = EarlyStop

class EXP_BASIC():
    early_stop: _EarlyStopType | None = None
    result_path: str | Path
    epochs: int
    wandb_logger: Run
    model: BaseModule | DDP

    def __init__(self, model, config, early_stop: _EarlyStopType | None = None, epochs=500, exp_id=os.environ['EXP_ID'], run_name: str="",
                 result_path: Path | str=os.environ['RES_PATH'], device=os.environ['ACCELERATOR'], model_path: Path|str = "") -> None:
        self.exp_id = exp_id
        self.run_name = run_name
        self.result_path = Path(result_path or \
            get_result_path(self.exp_id, self.run_name))
        self.logger = setup_logger(self.exp_id, self.result_path)
        self.logger.info("Initializing experiment...")
        self.logger.info("Checking device...")
        device_id = os.environ.get("LOCAL_RANK","0")
        self.device = torch.device(f"cuda:{device_id}") if device != 'cpu' else torch.device("cpu")
        if model_path:
            self.logger.info(f"Loading model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.logger.info(f"Model loaded to device: {self.device}")
        self.model = DDP(model.to(self.device), device_ids=[self.device], find_unused_parameters=True) if device == 'torchrun' else model.to(self.device)
        
        self.wandb_logger = wandb.init(
            project=self.exp_id,
            name=self.run_name,
            config=config,
            note=os.environ.get('EXP_NOTE',''),
            dir=str(self.result_path)
        )
        # hyper parameters
        self.early_stop = early_stop
        self.epochs = epochs
        self.model_path = model_path
        self.config = config
        self.step = 0

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_logger()
    
    def exit_logger(self):
        if self.wandb_logger is not None:
            self.wandb_logger.finish()

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
