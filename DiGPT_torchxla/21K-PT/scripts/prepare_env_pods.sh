# optional
pip install -U pip  # Yes, really needed.
  # NOTE: doesn't work when in requirements.txt -> cyclic dep
pip install tensorflow-cpu==2.11.0 tensorflow-datasets==4.8.2
pip install protobuf==3.19.6
pip install tensorflow-metadata==1.12
pip install timm==0.4.5 einops wandb==0.15.0
pip install tensorboardX==2.4
pip install open_clip_torch

pip install ftfy regex tqdm