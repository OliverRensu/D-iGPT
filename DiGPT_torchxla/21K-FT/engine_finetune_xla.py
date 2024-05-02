import logging

import torch
import torch_xla.core.xla_model as xm
import torch_xla.test.test_utils as test_utils
import torchvision.transforms as transforms
from util import misc
from util import lr_sched
from typing import Iterable, Optional

from timm.data import Mixup
from timm.utils import accuracy
import numpy as np
from util.precision import get_autocast
import math
import sys
import time

# import wandb

def after_train_step(args, times, data_iter_step, epoch, num_batches_per_epoch,
                     losses, lr):
    # NOTE loss is coarsely sampled, just master node and per log update
    if args.rank == 0 and (data_iter_step % args.log_freq or data_iter_step == num_batches_per_epoch) == 0:
        loss = losses.item()*args.accum_iter

        percent_complete = 100.0 * data_iter_step // args.accum_iter / num_batches_per_epoch
        samples_per_second = args.accum_iter * args.batch_size * args.world_size /times
        loss_log = f"loss: {loss}"
        logging.info(
            f"Train Epoch: {epoch} ({percent_complete:.0f}%)] "
            f"Batch (t): {times:.3f}, {samples_per_second:#g}/s"
            f"LR: {lr:7f} "  + loss_log
        )

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, writer,
                    mixup_fn=None, args=None):
    model.train(True)

    accum_iter = args.accum_iter
    autocast = get_autocast()
    optimizer.zero_grad()
    total_train_samples, total_train_batches, total_train_loss, total_train_lr = 0, 0, 0.0, 0.0
    num_batches_per_epoch = len(data_loader) // args.accum_iter
    #batch_time_m = AverageMeter()
    end = time.time()
    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (samples, targets) in enumerate(data_loader):
        if data_iter_step % accum_iter == 0:
            if data_iter_step%200==0:
                middle_step = data_iter_step//200*200 + 100
                lr_sched.adjust_learning_rate(optimizer, middle_step / len(data_loader) + epoch, args)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        samples = samples.to(device)
        targets = targets.to(device)
        with autocast():
            outputs = model(samples)
        loss = criterion(outputs, targets)
        loss /= accum_iter
        lr = optimizer.param_groups[-1]["lr"]

        total_train_loss += loss.item()
        total_train_lr += lr
        total_train_batches += 1
        loss.backward()
        if (data_iter_step + 1) % accum_iter != 0:
            xm.mark_step()
        else:
            #xm.reduce_gradients(optimizer)
            optimizer.step()
            xm.mark_step()
            optimizer.zero_grad()
        #batch_time_m.update(time.time() - end)

        after_train_step_args = [args, time.time() - end, data_iter_step, epoch, num_batches_per_epoch, loss, lr]
        xm.add_step_closure(after_train_step, after_train_step_args)
        end = time.time()
    train_loss = total_train_loss / total_train_batches
    train_lr = total_train_lr / total_train_batches
    train_loss = xm.mesh_reduce('train_loss', train_loss, np.mean)
    train_lr = xm.mesh_reduce('train_lr', train_lr, np.mean)

    return {"train_loss":train_loss, "train_lr":train_lr}


@torch.no_grad()
def evaluate(data_loader, model, epoch, device, mask_num=None, num_patches=None):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    autocast = get_autocast()
    total_test_samples, total_test_batches, total_test_loss, total_test_acc1, total_test_acc5 = 0, 0, 0.0, 0.0, 0.0
    for images, target in data_loader:
        images = images.to(device)
        target = target.to(device)
        with autocast(), torch.no_grad():
            output = model(images)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        total_test_loss += loss
        total_test_acc1 += acc1 * batch_size
        total_test_acc5 += acc5 * batch_size
        total_test_batches += 1
        total_test_samples += batch_size
        # test_acc1 = total_test_acc1.item() / total_test_samples
        # test_acc5 = total_test_acc5.item() / total_test_samples
        # test_loss = total_test_loss.item() / total_test_batches
        # logging.info("Loss: {}, Top-1: {}, Top-5: {}".format(test_loss, test_acc1, test_acc5))

    # gather
    # metric_logger.synchronize_between_processes()
    test_acc1 = total_test_acc1.item() / total_test_samples
    test_acc5 = total_test_acc5.item() / total_test_samples
    test_loss = total_test_loss.item() / total_test_batches
    logging.info("Loss: {}, Top-1: {}, Top-5: {}".format(test_loss, test_acc1, test_acc5))
    xm.add_step_closure(test_utils.print_test_update, args=(device, epoch, test_acc1))
    test_acc1 = xm.mesh_reduce('test_acc1', test_acc1, np.mean)
    test_acc5 = xm.mesh_reduce('test_acc5', test_acc5, np.mean)
    test_loss = xm.mesh_reduce('test_loss', test_loss, np.mean)

    if misc.is_main_process():
        logging.info('* Acc@1 {top1:.3f} Acc@5 {top5:.3f} loss {losses:.3f}'
              .format(top1=test_acc1, top5=test_acc5, losses=test_loss))
    return test_acc1, test_loss, test_acc5

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}