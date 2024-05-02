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
from typing import Iterable
import time
import torch
import torchvision.transforms as transforms
import numpy as np
import logging

import util.misc as misc
import util.lr_sched as lr_sched
from util.device_env_factory import use_xla
from util.precision import get_autocast

import torch_xla.core.xla_model as xm
import torch_xla
_HAS_XLA = True

try:
    import wandb
except ImportError:
    wandb = None


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def after_train_step(args, batch_time_m, data_iter_step, epoch, num_batches_per_epoch,
                     losses, lr, ar_losses, kd_losses):
    # NOTE loss is coarsely sampled, just master node and per log update
    if args.rank == 0 and (data_iter_step % args.log_freq or data_iter_step == num_batches_per_epoch) == 0:
        loss = (sum(losses) / len(losses)).item()
        ar_loss = (sum(ar_losses) / len(ar_losses)).item()
        kd_loss = (sum(kd_losses) / len(kd_losses)).item()

        percent_complete = 100.0 * data_iter_step // args.accum_iter / num_batches_per_epoch
        samples_per_second = args.accum_iter * args.batch_size * args.world_size / batch_time_m.val
        loss_log = f"loss: {loss} ar_loss: {ar_loss} kd_loss: {kd_loss}"
        logging.info(
            f"Train Epoch: {epoch} ({percent_complete:.0f}%)] "
            f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s"
            f"LR: {lr:7f} "  + loss_log
        )

        if args.wandb:
            assert wandb is not None, 'Please install wandb.'
            step = num_batches_per_epoch * epoch + data_iter_step // args.accum_iter
            wandb.log({'train_loss': loss}, step=step)
            wandb.log({'ar_loss': ar_loss}, step=step)
            wandb.log({'kd_loss': kd_loss}, step=step)
            wandb.log({'lr': lr}, step=step)

        batch_time_m.reset()


def train_one_epoch(model: torch.nn.Module, teacher,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    # log_writer=None,
                    args=None):
    model.train(True)

    autocast = get_autocast()

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    batch_time_m = AverageMeter()
    end = time.time()
    losses = []
    ar_losses = []
    kd_losses = []
    total_loss = total_ar_seg2 = total_ar_seg3 = total_ar_seg4 = total_kd_seg2 = total_kd_seg3 = total_kd_seg4 = total_lr = 0
    num_batches_per_epoch = len(data_loader) // args.accum_iter

    for data_iter_step, (samples, _) in enumerate(data_loader):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        student_samples = samples[0].to(device, non_blocking=True)
        teacher_samples = samples[1].to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        with autocast():
            with torch.no_grad():
                teacher_out = teacher(teacher_samples)
            ar_loss, kd_loss = model(student_samples, teacher_out)
            loss = ar_loss.mean() + kd_loss.mean()

        loss /= accum_iter
        losses.append(loss)
        ar_losses.append(ar_loss)
        kd_losses.append(kd_loss)
        # if use_xla():
        loss.backward()
        if (data_iter_step + 1) % accum_iter != 0:
            xm.mark_step()
        else:
            #xm.reduce_gradients(optimizer)
            optimizer.step()
            xm.mark_step()
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]["lr"]
        total_loss += loss * accum_iter
        total_ar_seg2 += ar_loss
        total_kd_seg2 += kd_loss
        total_lr += lr

        batch_time_m.update(time.time() - end)
        end = time.time()
        after_train_step_args = [args, batch_time_m, data_iter_step, epoch, num_batches_per_epoch, losses, lr, ar_losses, kd_losses]
        xm.add_step_closure(after_train_step, after_train_step_args)
        if (data_iter_step + 1) % accum_iter == 0:
            losses = []
            ar_losses = []
            kd_losses = []
    return_dict = dict(
        loss=total_loss.item() / len(data_loader),
        ar_seg2=total_ar_seg2.item() / len(data_loader),
        kd_seg2=total_kd_seg2.item() / len(data_loader),
        lr=total_lr / len(data_loader),
    )
    new_return_dict = dict()
    for key, val in return_dict.items():
        val = xm.mesh_reduce(key, val, np.mean)
        new_return_dict[key] = val
    return new_return_dict