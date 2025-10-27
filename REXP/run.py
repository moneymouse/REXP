import argparse
import tomllib
import os
import sys
import runpy
from utils.tools import AttrDict, setup_logger
import torch
from utils.tools import get_result_path, HiddenPrints

parse = argparse.ArgumentParser()
sys.path.insert(0, os.getcwd())

# training settings are decieded by the experiment id
parse.add_argument("-exp_id", "--experiment_id", help="The experiment id used to identify the experiment. \n " \
                    "Recommend to set EXP_ID environmental variable.", type=str)
parse.add_argument("exp_path", help="EXP script path to run.", type=str)
parse.add_argument("-a", "--accelerator", choices=['cpu','gpu','torchrunn'], default='torchrun', 
                   help="Choose your expected devices to train or reason your model. " \
                        "Please use 'CUDA_VISIBLE_DEVICES' enviromental variable to control the GPU id used to train. " \
                        "When gpu was chosen, the visible gpu with smallest order will be used.")
parse.add_argument("-p", "--result_path", type=str, default="", required=False, help="The dir path of results. Default in `res/exp_id` dir.")
parse.add_argument("--debug", action="store_true", default=False, required=False, help="'on' or 'off': turn on or off debug mode.")
parse.add_argument("-n","--title", type=str, default="", required=False, help="A brief note to describe this experiment.")

def main():
     # with HiddenPrints(int(os.environ.get("LOCAL_RANK", '0'))):
     args = AttrDict()
     parse.parse_args(namespace=args)
     # set the experiment id
     args.experiment_id = os.environ.get("EXP_ID", args.experiment_id)
     # setup exp environment
     os.environ['EXP_ID'] = args.experiment_id
     os.environ['ACCELERATOR'] = args.accelerator
     if args.experiment_id is None:
          raise ValueError("Please set the experiment id by -exp or EXP_ID enviromental variable.")
     # exp config
     exponfig = {}
     with open("./exponfig.toml", "rb") as fp:
          exponfig = tomllib.load(fp)
          for k,v in exponfig.get("env", {}).items():
               os.environ[k] = str(v)
          
     # os.environ['WANDB_ENTITY'] = exponfig.get("entity", '')
     os.environ['DEBUG'] = "on" if args.debug else os.environ.get("DEBUG", 'off')
     
     if args.accelerator == 'torchrun':
          torch.distributed.init_process_group(backend='nccl', init_method="env://")
     
     # generate res path and make dir in res path.
     res_path = get_result_path(args.experiment_id, args.title) if not args.result_path else args.result_path
     os.environ['RES_PATH'] = str(res_path)
     os.environ['EXP_NOTE'] = args.title if args.title else "" # record note for the later use
     
     # setup logger
     logger = setup_logger(args.experiment_id, res_path)
     
     with HiddenPrints(os.environ.get('LOCAL_RANK','0')):
          logger.info("Start experiment: %s", args.experiment_id)
          runpy.run_path(args.exp_path, run_name="__main__")
          logger.info("Experiment finished.")
          # log save_dir of the experiment
          logger.info("Experiment save_dir: %s", os.environ['RES_PATH'])
     torch.distributed.destroy_process_group() if args.accelerator == 'torchrun' else None

if __name__ == '__main__':
     main()