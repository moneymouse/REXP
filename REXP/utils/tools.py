import logging
import datetime
import os
from pathlib import Path
from sympy import true
import torch
import numpy as np
import pickle
import shutil
import sys
import wandb

class AttrDict(dict):
    """A class that allows you to access dictionary keys as attributes."""
    def __getattr__(self, item):
        if item in self:
            return self[item]
        raise AttributeError(f"'AttrDict' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

class HiddenPrints:
    def __init__(self, rank):
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if str(self.rank) == '0':
            return
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if str(self.rank) == '0':
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class RankFilter(logging.Filter):
    def __init__(self, rank=0):
        super().__init__()
        self.threshold_rank = rank

    def filter(self, record):
        # True 才会输出
        current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return current_rank == self.threshold_rank

def setup_logger(experiment_id, result_path: Path, log_level=logging.INFO):
    """
    设置日志记录器，支持同时输出到文件和控制台。
    
    :param experiment_id: 实验ID，用于标识日志
    :param result_path: 日志文件保存路径
    :param log_level: 日志级别，默认为INFO
    :return: 配置好的日志记录器
    """
    init_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    result_path.mkdir(parents=True, exist_ok=True)

    streamHandler = logging.StreamHandler()
    streamHandler.addFilter(logging.Filter(os.environ['EXP_ID']))
    streamHandler.addFilter(RankFilter())
    
    fileHandler = logging.FileHandler(Path(result_path) / f"{experiment_id}__{init_time}.log")
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[fileHandler, streamHandler]
    )
    
    return logging.getLogger(experiment_id)

class EarlyStop:
    def __init__(self, path: str | Path = None, patience=2, save_model=True, ddp=False):
        r"""
            Args:
                path:
                    Default `os.environ[RES_PATH]`. The parent path of the model's parameters.
        """
        self.patience = patience
        self.path = Path(path or os.environ.get('RES_PATH'))
        # self.validation_loss = -1
        self.count = 0
        self.best_score = np.inf
        self.save = save_model
        self.ddp = ddp
        self.rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    def __call__(self, model, vali_score) -> bool:
        if vali_score > self.best_score:
            self.count += 1
            logger = logging.getLogger(__name__)
            logger.info(f'EarlyStopping counter: {self.count} out of {self.patience}')
            if self.count >= self.patience:
                return True
        else:
            self.count = 0
            self.best_score = vali_score
            self._save_model(model)
        return False
    
    def _save_model(self, model):
        if self.save and self.rank == 0:
            torch.save(model.state_dict(), self.path / "checkpoint.pth")
        # self.logger.info("Best model saved at {}".format(self.path))
    
    def load_model(self, model):
        if self.save:
            state_dict = torch.load(self.path / "checkpoint.pth")
            model.load_state_dict(state_dict)
            return model

    def refresh(self):
        r""" Calm down the early stop for model training/finetuning in new dataset. """
        self.count = 0
        self.best_score = np.inf


''' return directory path of result '''
def get_result_path(exp_id, title="") -> Path:
    root = Path("./res")
    exp_path = root / exp_id
    # mkdir
    Path.mkdir(exp_path, exist_ok=True)
    version = torch.empty(1)

    if os.environ['ACCELERATOR'] != 'torchrun' or os.environ['LOCAL_RANK'] == 0: 
        _version = -1
        if (Path.exists(exp_path / ".version")):
            with open(exp_path / ".version", "rb") as fp:
                _version = int(pickle.load(fp).get("version", 0))

        with open(exp_path / ".version", "wb") as fp:
            pickle.dump({"version": _version + 1}, fp)
        
        version.fill_(_version)
    
    # sync version
    if os.environ['ACCELERATOR'] == 'torchrun': 
        version = version.to(f"cuda:{os.environ['LOCAL_RANK']}")
        torch.distributed.broadcast(version, src=0)
    
    version = version.int().item()
    res = (exp_path / f"{version}.{title}") if title else (exp_path / f"{version}")
    os.makedirs(res, exist_ok=True)
    return res

def moveto(n):
    # 受[tqdm](tqdm.tqdm.moveto)启发, 命令行行位移算法
    print("\n" * n + "\x1b[A" * (-n), flush=true, end="")

def pos_print(pos:int, values:str):
    # logging into res file.
    logger = logging.getLogger("EXP_RECORD")
    # remove the handler
    logger.info(values)

    # 在命令行任意行打印，pos=0为当前位置
    moveto(pos)
    n = shutil.get_terminal_size().columns
    print("\r" + values, end="", flush=True)
    # 移动到行尾
    print(f"\x1b[{n}G", end="", flush=True)
    moveto(-pos)

class WandbContext:
    """
    A context manager for wandb initialization that respects the DEBUG environment variable.
    When DEBUG is set, wandb initialization is skipped.
    
    Usage:
        with WandbContext(project="my_project", entity="my_entity") as run:
            # run will be None if DEBUG is set
            if run:
                run.log({"metric": value})
    """
    def __init__(self, **wandb_init_kwargs):
        """
        Initialize the wandb context manager with wandb.init kwargs
        
        Args:
            **wandb_init_kwargs: Arguments to pass to wandb.init
        """
        self.wandb_init_kwargs = wandb_init_kwargs
        
        if('project' not in self.wandb_init_kwargs):
            self.wandb_init_kwargs['project'] = os.environ.get('EXP_ID', 'default_project')

        self.run = None
        self.debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "on")
        
    def __enter__(self):        
        # Check if RES_PATH exists and rename it if needed
        res_path = os.environ.get('RES_PATH')
        if res_path and os.path.exists(res_path):
            # Construct the new name using wandb project or name if available
            res_path = Path(res_path)
            prefix = self.wandb_init_kwargs.get('name', '')
            
            if prefix:
                new_path = res_path.parent / f"{prefix}_{res_path.name}"
                try:
                    os.environ['NEW_PATH'] = str(new_path)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to rename result path: {e}")

        # if not self.debug_mode and os.environ.get("LOCAL_RANK", "0") == "0":
        try:
            self.run = wandb.init(**self.wandb_init_kwargs, notes=os.environ.get("EXP_NOTE", ""), mode='online' 
                                  if not self.debug_mode and os.environ.get("LOCAL_RANK", "0") == "0" else 'disabled')
        except ImportError:
            logger = logging.getLogger(__name__)
            logger.warning("wandb not installed. Running without wandb tracking.")
        return self.run
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run is not None:
            self.run.finish()

    @staticmethod
    def update_dir():
        if new_path == os.environ.get("NEW_PATH", False):
            res_path = Path(os.environ.get("RES_PATH"))
            new_path = Path(new_path)
            res_path.rename(new_path)

# 扩展导出
__all__ = [
    'AttrDict',
    'HiddenPrints',
    'setup_logger',
]
