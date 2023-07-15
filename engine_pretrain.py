# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable, Optional

import torch
import math
import random
import torch.nn.functional as F
from models_mae import AdversarialLoss, Wasserstein

from util.datasets import DataPrefetcher
import util.misc as misc
import util.lr_sched as lr_sched
from contextlib import nullcontext

def random_rectangle_simple(mask_factor, grid_size):
    area = random.randint(40, 160)
    l = random.randint(4, grid_size - 4)
    w = min(grid_size, math.ceil(area / l))
    x = random.randint(0, grid_size - l)
    y = random.randint(0, grid_size - w)
    mask = torch.zeros(grid_size, grid_size)
    mask[y:y+w, x:x+l] = mask_factor 
    mask += torch.rand_like(mask)
    return mask.flatten()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    scorer: Optional[torch.nn.Module]=None,
                    discriminator: Optional[torch.nn.Module]=None,
                    optimizer_d: Optional[torch.optim.Optimizer]=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    criterion = AdversarialLoss(type='lsgan').to(device)


    prefetcher = DataPrefetcher(data_loader)
    for data_iter_step, batch in enumerate(metric_logger.log_every(prefetcher, print_freq, header)):
        if len(batch) == 3:
            samples, _, masks = batch
            # samples_o, _, masks_o = batch_o
            masks = masks.to(device, non_blocking=True)
            # masks_o = masks_o.to(device, non_blocking=True)
        else:
            samples, _ = batch
            # samples_o, _ = batch_o
            masks = None
            # masks_o = None

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(optimizer_d, data_iter_step / len(data_loader) + epoch, args)
        
        samples = samples.to(device, non_blocking=True)
        discriminator_reduce_ctx = discriminator.no_sync if misc.get_rank() != -1 and (data_iter_step + 1) % accum_iter != 0 else nullcontext
        with torch.no_grad():
            mask_tr = torch.zeros(samples.shape[0], args.num_patches)
            for i in range(samples.shape[0]):
                mask_true = random_rectangle_simple(0.5, args.grid_size[0])
                mask_tr[i] = mask_true
            mask_tr = mask_tr.to(device, non_blocking=True)
            attn_map = scorer.get_last_selfattention(samples)[:,:, 0, 1:]
            mask_prob_d = model(attn_map, mode='mask')
            mask_prob_d = mask_prob_d.reshape(-1, 1, args.grid_size[0], args.grid_size[1])
            mask_tr = F.softmax(mask_tr, dim=-1)
            mask_tr = mask_tr.reshape(-1, 1, args.grid_size[0], args.grid_size[1])

        with discriminator_reduce_ctx():
            loss_d1r = criterion(discriminator(mask_tr), is_real=True)
            loss_d1f = criterion(discriminator(mask_prob_d), is_real=False)
            loss_d_1 = loss_d1r + loss_d1f
            loss_d = loss_d_1
            loss_d /= accum_iter
            loss_d.backward()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_d.step()
            optimizer_d.zero_grad()

        for p in discriminator.parameters():
           p.requires_grad_(False)

        model_reduce_ctx = model.no_sync if misc.get_rank() != -1 and (data_iter_step + 1) % accum_iter != 0 else nullcontext
        with model_reduce_ctx():
            with torch.cuda.amp.autocast():
                loss, _, (masks_actual, mask_prob) = model(samples, mask_ratio=args.mask_ratio, mask=attn_map, mask_factor=args.mask_factor)

            loss_g = criterion(discriminator(mask_prob.reshape(-1, 1, args.grid_size[0], args.grid_size[1])), is_real=True)
            loss = loss + args.loss_g_factor * loss_g

            for p in discriminator.parameters():
                p.requires_grad_(True)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_d_reduce = misc.all_reduce_mean(loss_d.item())
        loss_d1r_reduce = misc.all_reduce_mean(loss_d1r.item())
        loss_d1f_reduce = misc.all_reduce_mean(loss_d1f.item())
        loss_g_reduce = misc.all_reduce_mean(loss_g.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('gan/d', loss_d_reduce, epoch_1000x)
            log_writer.add_scalar('gan/d1r', loss_d1r_reduce, epoch_1000x)
            log_writer.add_scalar('gan/d1f', loss_d1f_reduce, epoch_1000x)
            log_writer.add_scalar('gan/g', loss_g_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
