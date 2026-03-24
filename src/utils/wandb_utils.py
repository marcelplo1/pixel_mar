import wandb
import torch.distributed as dist
import os
import argparse
import hashlib

def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }

def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def initialize_wandb(args, entity, exp_name, project_name, model_config=None, sampler_config=None):
    config_dict = namespace_to_dict(args)
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_KEY"]
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )
    if model_config is not None:
        wandb.config.update({"model_config": model_config}, allow_val_change=True)
    if sampler_config is not None:
        wandb.config.update({"sampler_config": sampler_config}, allow_val_change=True)
    wandb.define_metric("epoch") 
    wandb.define_metric("evaluate/*", step_metric="epoch")

def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)
