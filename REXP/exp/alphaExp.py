from pyexpat import model
import time
from collections.abc import Callable
from typing import Protocol, TypeVar
from xml.dom import NotFoundErr
from pydantic import BaseModel
from torch.utils.data.dataloader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import torch
from tqdm import trange
import os
import numpy as np

from utils.tools import pos_print

from .exp_basic import EXP_BASIC

class LoopFn(Protocol):
    def __call__(self, model: BaseModel, idx: int, batch: list[torch.Tensor]) -> torch.Tensor:
        ...

T = TypeVar("T", bound=LoopFn)

logger = logging.getLogger(__name__)

class EXP(EXP_BASIC):
    loader: DataLoader
    vali_loader: None | DataLoader

    _training_loop: Callable[[BaseModel ,int, list[torch.Tensor]], torch.Tensor]
    _vali_loop: Callable[[BaseModel ,int, list[torch.Tensor]], torch.Tensor]
    _testing_loop: Callable[[BaseModel ,int, list[torch.Tensor]], torch.Tensor]

    def __call__(self, loader: DataLoader, vali_loader: None|DataLoader = None):
        self.loader = loader
        self.vali_loader = vali_loader
        self.train()
        return
    
    def training_step(self, func: Callable[[BaseModel ,int, list[torch.Tensor]], torch.Tensor]):
        self._training_loop = func
        return func
    
    def vali_step(self, func):
        self._vali_loop = func
        return func
    
    def testing_step(self, func: Callable[[BaseModel ,int, list[torch.Tensor]], torch.Tensor]):
        self._testing_loop = func
        return func
    
    def train(self):

        # Adam optimizer
        # TODO: it's necessary to give a lr to configure_optim. 
        # It's EXP's busness!
        # lr scheduler and optimizer
        if(type(self.model) == DDP):
            optimizer = self.model.module.configure_optim(self.config['lr']) # type: ignore
            scheduler = self.model.module.lr_schedulers() # type: ignore
        else:
            optimizer = self.model.configure_optim(self.config['lr'])
            scheduler = self.model.lr_schedulers() # type: ignore

        # train the model
        self.logger.info("Start training...")
        time_now = time.time()
        self.model.train()
        with trange(self.epochs, position=0, leave=True) as epoch_bar:
            # epoch
            for epoch in range(self.epochs):
                epoch_bar.set_description(f"Epoch")
                epoch_losses: list[torch.Tensor] = []
                with trange(len(self.loader), position=1, leave=False) as batch_bar:   
                    batch_bar.set_description("Batch") 
                    for idx, batch in enumerate(self.loader):
                        optimizer.zero_grad()
                        batch_tensor = [torch.Tensor(b).float().to(self.device) for b in batch]
                        loss = self._training_loop(self.model ,idx, batch_tensor)
                        # bp
                        # TODO: distributed training barrier?
                        loss.backward()
                        optimizer.step()
                        epoch_losses.append(loss)
                        if(idx % 50 == 0):
                            # wandb log
                            self.wandb_logger.log({'loss': loss.item()}, step=self.step) if self.wandb_logger != None else None
                            pos_print(2, "\rsteps: {6}, iters: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB"
                            .format(idx, epoch + 1, loss.item(), time.time() - time_now,
                            torch.cuda.memory_allocated() / 1024 / 1024,
                            torch.cuda.memory_reserved() / 1024 / 1024,
                            self.step))
                        self.step += 1
                        # torch.cuda.empty_cache()
                        batch_bar.update(10) if (idx % 10 == 0) else None
                # caculate and log the loss
                if(len(epoch_losses) == 0): continue
                train_loss =  torch.Tensor(epoch_losses).detach().mean().item() / len(epoch_losses)
                vali_loss = None
                if(self.vali_loader):
                    vali_loss = self.vali()
                    pos_print(3 ,f"train_loss: {train_loss:.4f}, vali_loss: {vali_loss:.4f}")
                    # epoch_bar.set_postfix({"train_loss": f"{train_loss:.4f}", "vali_loss": f"{vali_loss:.4f}"})
                    # with logging_redirect_tqdm():
                        # logger.info(f"Train loss: {train_loss:.4f} Validation loss {vali_loss:.4}")
                    self.wandb_logger.log({'train/loss': train_loss, 'vali/loss': vali_loss}, step=self.step) if self.wandb_logger != None else None
                else:
                    pos_print(3 ,f"\ttrain_loss: {train_loss:.4f}")
                    # with logging_redirect_tqdm():
                    #     logger.info(f"Train loss: {train_loss:.4f}")
                    self.wandb_logger.log({'train/loss': train_loss}, step=self.step) if self.wandb_logger != None else None
                
                # early stop and save best model
                if os.environ.get('LOCAL_RANK', '0') == '0':
                    if vali_loss and self.early_stop and self.early_stop(self.model, vali_loss):
                        break
                
                # schedule
                if type(scheduler) == list:
                    [s.step(epoch) for s in scheduler]
                if type(scheduler) == torch.optim.lr_scheduler.LRScheduler:
                    scheduler.step(epoch)
                
                # update the progress bar
                epoch_bar.update(1)
                
        self.logger.info("Best model saved at {}".format(self.early_stop.path)) if self.early_stop and self.early_stop.save else None
        

    def vali(self):
        self.model.eval()
        losses:list[torch.Tensor] = []
        if not self.vali_loader:
            raise NotFoundErr("validate dataset not found")
        with torch.no_grad():
            with trange(len(self.vali_loader),leave=False) as vali_bar:
                vali_bar.set_description("Validating")
                for idx, batch in enumerate(self.vali_loader):
                    batch_tensor = [torch.Tensor(b).float().to(self.device) for b in batch]
                    loss = self._vali_loop(self.model, idx, batch_tensor)
                    losses.append(loss)
                    vali_bar.update(1)
                    # torch.cuda.empty_cache()
            
        vali_loss = sum(losses).item() / len(losses)
        return np.float64(vali_loss)
        
    def test(self, test_loader: DataLoader | None, model: BaseModel | None = None):
        # load best model after training
        best_model = self.early_stop.load_model(self.model) if self.early_stop and self.early_stop.save else self.model

        _model = model or best_model        
        _model.eval()

        self.wandb_logger.define_metric("test/*", step_metric="test/steps") if self.wandb_logger != None else None

        if not test_loader:
            raise NotFoundErr("test_loader dataset not found")
        
        losses: list[torch.Tensor] = []
        
        with torch.no_grad():
            with trange(len(test_loader), position=0, leave=False) as test_bar:
                for idx, batch in enumerate(test_loader):
                    test_bar.set_description("Testing")
                    batch_tensor = [torch.Tensor(b).float().to(self.device) for b in batch]
                    loss = self._testing_loop(_model, idx, batch_tensor)
                    losses.append(loss)

                    self.wandb_logger.log({
                        "test/steps": idx,
                        "test/loss": loss.detach().item()
                    }) if self.wandb_logger != None else None
                    test_bar.update(10) if idx % 10 == 0 else None
                    # torch.cuda.empty_cache()
        
        avg_loss = sum(losses).detach().item() / len(losses)
        logger.info(f"test loss: {avg_loss}")