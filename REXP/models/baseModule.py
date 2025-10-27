from ast import Not
from unittest.mock import Base
from torch import nn, optim
from typing import Union

import torch

class BaseModule(nn.Module):
    def configure_optim(self, lr=0.0001) -> optim.Optimizer:
        raise NotImplementedError

    def lr_schedulers(self) -> Union[None, list[Union[optim.lr_scheduler.LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]], optim.lr_scheduler.LRScheduler, optim.lr_scheduler.ReduceLROnPlateau]:
        pass