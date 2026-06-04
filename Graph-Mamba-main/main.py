import datetime
import os
import torch
import logging
import warnings

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_agg_dir, set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
from graphgps.logger import create_logger


def new_optimizer_config(cfg):
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum
    )


def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period
    )


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    if len(cfg.run_multiple_splits) == 0:
        # multi-seed mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # multi-split mode
        if args.repeat != 1:
            raise NotImplementedError("Multiple repeats of multiple splits not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # Load command line args
    args = parse_args()
    # Load config
    set_cfg(cfg)
    load_cfg(cfg, args)
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    # -----------------------------
    # 自动选择 device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"[INFO] Using device: {device}")
    cfg.device = str(device)

    # Set Pytorch threads
    torch.set_num_threads(cfg.num_threads)

    # Repeat experiments
    for run_id, seed, split_index in zip(*run_loop_settings()):
        # Setup directories & config
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()

        # Pretrained model config
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)

        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")

        # Loaders & loggers
        loaders = create_loader()
        loggers = create_logger()
        if cfg.dataset.name in ['ogbn-arxiv', 'ogbn-proteins']:
            loggers_2 = create_logger()
            loggers_3 = create_logger()
            loggers_2[0].name = "val"
            loggers_3[0].name = "test"
            loggers.extend(loggers_2)
            loggers.extend(loggers_3)
            loaders = loaders * 3

        # Model
        model = create_model().to(device)
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model,
                cfg.pretrained.dir,
                cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head
            )

        # Optimizer & scheduler
        optimizer = create_optimizer(model.parameters(), new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

        # Logging model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)

        # Train
        if cfg.train.mode == 'standard':
            if cfg.wandb.use:
                logging.warning("[W] WandB logging not supported in standard mode. Set train.mode='custom'")
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    # Aggregate results
    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as e:
        logging.info(f"Failed to aggregate multiple runs: {e}")

    # Mark yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')

    logging.info(f"[*] All done: {datetime.datetime.now()}")
