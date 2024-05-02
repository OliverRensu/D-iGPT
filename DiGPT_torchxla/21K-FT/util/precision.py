import torch
from contextlib import suppress

from .device_env_factory import use_xla

def get_autocast():
    if use_xla():
        return suppress
    else:
        return torch.cuda.amp.autocast
