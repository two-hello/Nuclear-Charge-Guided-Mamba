import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from graphgps import loader
from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name

from deepspeed.profiling.flops_profiler import FlopsProfiler
from torch.profiler import profile, record_function, ProfilerActivity


def subsample_batch_index(batch, min_k=1, ratio=0.1):
    torch.manual_seed(0)
    unique_batches = torch.unique(batch.batch)
    permuted_indices = []
    for batch_index in unique_batches:
        indices_in_batch = (batch.batch == batch_index).nonzero().squeeze()
        k = int(indices_in_batch.size(0) * ratio)
        if k > min_k:
            perm = torch.randperm(indices_in_batch.size(0))
            idx = perm[:k]
            idx = indices_in_batch[idx]
            idx, _ = torch.sort(idx)
        else:
            idx = indices_in_batch
        permuted_indices.append(idx)
    idx = torch.cat(permuted_indices)
    return idx


def arxiv_cross_entropy(pred, true, split_idx):
    true = true.squeeze(-1)
    if pred.ndim > 1 and true.ndim == 1:
        pred_score = F.log_softmax(pred[split_idx], dim=-1)
        loss = F.nll_loss(pred_score, true[split_idx])
    else:
        raise ValueError("In ogbn cross_entropy calculation dimensions did not match.")
    return loss, pred_score


def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    if_mem = False
    if_flop = False
    if_select = False
    if if_flop:
        prof = FlopsProfiler(model, None)
        total_flop_s = 0.
        sample_count = 0
        if if_select:
            total_node = 0

    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        if if_select:
            ratio = 1.0
            idx = subsample_batch_index(batch, min_k=1, ratio=ratio)
            batch = batch.subgraph(idx)
        if if_flop:
            prof.start_profile()
        batch.split = 'train'
        batch.to(torch.device(cfg.device))

        pred, true = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'ogbn-arxiv':
            split_idx = loader.dataset.split_idx['train'].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, split_idx)
            _true = true[split_idx].detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)

        if if_flop:
            prof.stop_profile()
            flops = prof.get_total_flops()
            flops_s = flops / 1000000000.
            total_flop_s += flops_s
            sample_count += len(torch.unique(batch.batch))
            params = prof.get_total_params()
            prof.end_profile()
            if if_select:
                total_node += batch.x.size(0)

        loss.backward()
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true, pred=_pred, loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0], time_used=time.time() - time_start,
                            params=None, dataset_name=cfg.dataset.name)
        time_start = time.time()
    if if_flop:
        print('################ Print flop')
        print(total_flop_s / sample_count, params)
        print('################ End print flop')
    if if_mem:
        print('################ Print mem')
        print(torch.cuda.max_memory_allocated() / (1024 ** 2))
        print(torch.cuda.max_memory_reserved() / (1024 ** 2))
        print('################ End print mem')
    if if_select:
        print('################ Print avg nodes')
        print(total_node / sample_count)


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        elif cfg.dataset.name == 'ogbn-arxiv':
            index_split = loader.dataset.split_idx[split].to(torch.device(cfg.device))
            loss, pred_score = arxiv_cross_entropy(pred, true, index_split)
            _true = true[index_split].detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true, pred=_pred, loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=None, dataset_name=cfg.dataset.name, **extra_stats)
        time_start = time.time()


# ========== 原子序数 Z 重要性分析函数 (绝杀版) ==========
@torch.no_grad()
@torch.no_grad()
def evaluate_z_importance(model, loader, device):
    """
    🏆 完美版：增加相对百分比计算
    """
    model.eval()
    total_loss_base = 0.0
    total_loss_shuffle = 0.0
    num_batches = 0

    logging.info("=" * 60)
    logging.info("开始执行：固定Position，Shuffle Z 对照实验")
    logging.info("实验规则：仅打乱原子序数Z，图结构/位置/其他特征完全不变")
    logging.info("=" * 60)

    for batch in loader:
        batch = batch.to(device)
        batch.split = 'val'
        num_batches += 1

        # ---------------------------------------------------------
        # 1. 全量备份 (在任何前向之前)
        # ---------------------------------------------------------
        backup = {}
        for key in batch.keys:
            val = batch[key]
            if isinstance(val, torch.Tensor):
                backup[key] = val.clone()

        # ---------------------------------------------------------
        # 2. 原始前向 (Baseline)
        # ---------------------------------------------------------
        pred_original, true_original = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss_base, _ = subtoken_cross_entropy(pred_original, true_original)
        elif cfg.dataset.name == 'ogbn-arxiv':
            index_split = loader.dataset.split_idx['val'].to(device)
            loss_base, _ = arxiv_cross_entropy(pred_original, true_original, index_split)
        else:
            loss_base, _ = compute_loss(pred_original, true_original)
        total_loss_base += loss_base.item()

        # ---------------------------------------------------------
        # 3. 全量恢复 + 打乱 Z
        # ---------------------------------------------------------
        # 先把所有东西恢复原状
        for key, val in backup.items():
            batch[key] = val

        # 现在 batch 是全新的了，开始打乱
        # 打乱 initial_x
        z = batch.initial_x[:, 0].clone()
        for graph_id in torch.unique(batch.batch):
            mask = batch.batch == graph_id
            z_g = z[mask]
            perm = torch.randperm(z_g.size(0), device=device)
            z[mask] = z_g[perm]

        # 替换 initial_x
        initial_x_shuffled = batch.initial_x.clone()
        initial_x_shuffled[:, 0] = z
        batch.initial_x = initial_x_shuffled

        # 同时替换 x，确保 Encoder 收到 Long 类型
        batch.x = initial_x_shuffled

        # ---------------------------------------------------------
        # 4. 打乱后前向
        # ---------------------------------------------------------
        pred_shuffle, true_shuffle = model(batch)
        if cfg.dataset.name == 'ogbg-code2':
            loss_shuffle, _ = subtoken_cross_entropy(pred_shuffle, true_shuffle)
        elif cfg.dataset.name == 'ogbn-arxiv':
            loss_shuffle, _ = arxiv_cross_entropy(pred_shuffle, true_shuffle, index_split)
        else:
            loss_shuffle, _ = compute_loss(pred_shuffle, true_shuffle)
        total_loss_shuffle += loss_shuffle.item()

    # ---------------------------------------------------------
    # 5. 计算结果
    # ---------------------------------------------------------
    avg_loss_base = total_loss_base / num_batches
    avg_loss_shuffle = total_loss_shuffle / num_batches
    z_importance_abs = avg_loss_shuffle - avg_loss_base

    # 🆕 计算相对百分比变化
    z_importance_rel_pct = (z_importance_abs / avg_loss_base) * 100

    logging.info("📊 实验结果汇总")
    logging.info(f"原始平均损失 (Baseline)：{avg_loss_base:.6f}")
    logging.info(f"Z打乱后平均损失：{avg_loss_shuffle:.6f}")
    logging.info(f"👉 原子序数Z重要性 (绝对差值)：{z_importance_abs:.6f}")
    logging.info(f"👉 原子序数Z重要性 (相对提升)：{z_importance_rel_pct:.2f}%")
    logging.info(f"结论：相对百分比越高 → 模型越依赖Z的化学语义")
    logging.info("=" * 60)

    result = {
        "avg_loss_baseline": avg_loss_base,
        "avg_loss_z_shuffled": avg_loss_shuffle,
        "z_importance_absolute": z_importance_abs,
        "z_importance_relative_percent": z_importance_rel_pct
    }
    torch.save(result, os.path.join(cfg.run_dir, "z_importance_experiment_result.pt"))
    return result

@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        import wandb
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]), cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"
                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and best_epoch == cur_epoch:
                save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: gamma={gtl.attention.gamma.item()}")

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")

    # ========== 训练结束后自动执行 Z-importance 分析 ==========
    logging.info("Starting automatic Z-importance analysis (Fixed Position, Shuffle Z)...")
    try:
        z_result = evaluate_z_importance(model, loaders[1], cfg.device)
        logging.info(f"Z-importance analysis completed. Results saved to {cfg.run_dir}")
    except Exception as e:
        logging.error(f"Z-importance analysis failed: {e}")
        import traceback
        traceback.print_exc()

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)