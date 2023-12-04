
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, teacher,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    newmean = [0.48145466, 0.4578275, 0.40821073]
    newstd = [0.26862954, 0.26130258, 0.27577711]
    mean = torch.as_tensor(mean).to(device)[None, :, None, None]
    std = torch.as_tensor(std).to(device)[None, :, None, None]
    newmean = torch.as_tensor(newmean).to(device)[None, :, None, None]
    newstd = torch.as_tensor(newstd).to(device)[None, :, None, None]

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                samples_T = samples * std + mean
                teacher_out = teacher((samples_T-newmean)/newstd)
            if (data_iter_step + 1) % accum_iter != 0:
                with model.no_sync():
                    ar_loss, kd_loss = model(samples, teacher_out)
                    loss = ar_loss + kd_loss
                    loss /= accum_iter
                    loss_scaler(loss, optimizer, parameters=model.parameters(),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
            else:
                ar_loss, kd_loss = model(samples, teacher_out)
                loss = ar_loss + kd_loss
                loss /= accum_iter
                loss_scaler(loss, optimizer, parameters=model.parameters(),
                            update_grad=(data_iter_step + 1) % accum_iter == 0)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(ar_seg2=ar_loss.item())
        metric_logger.update(kd_seg2=kd_loss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}