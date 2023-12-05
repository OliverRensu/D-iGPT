# D-iGPT

This repository is the official  PyTorch+GPU implementation of our 

**Rejuvenating image-GPT as Strong Visual Representation Learners**

*[Sucheng Ren](https://oliverrensu.github.io/), [Zeyu Wang](https://zw615.github.io/), [Hongru Zhu](https://pages.jh.edu/hzhu38/) [Junfei Xiao](https://lambert-x.github.io/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/), [Cihang Xie](cihangxie.github.io)*



## 🛠 Installation
We build the repo based on [MAE](https://github.com/facebookresearch/mae)

## 🚀 Pretraining
We pretrain TinyMIM on 32 A5000 GPU with overall batch size of 4096 which is identical to that in MAE.
```
python -m torch.distributed.launch \
--nnodes 4 --node_rank $noderank \
--nproc_per_node 8 --master_addr $ip --master_port $port \
main_pretrain.py \
    --batch_size 64 --accum_iter 2 \
    --model mae_vit_base_patch16 \
    --clip_path /path/to/openclip_vit_h_14.pth \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /path/to/ImageNet/
```
If your GPU has enough memory, you can set batch_size=128 accum_iter=1

## Fine-tuning on ImageNet-1K (Classification)
```
python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --batch_size 128 \
    --model vit_base \
    --finetune /path/to/checkpoint-299.pth \
    --epochs 100 \
    --output_dir ./out_finetune/ \
    --blr 1e-4 --layer_decay 0.6 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path /path/to/ImageNet/
```


The torch+GPU code produces better results. This is likely caused by the system difference between torch+GPU and torchxla+TPU.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Base</th>
<!-- TABLE BODY -->
<tr><td align="left">torch+GPU</td>
<td align="center">86.2</td>
</tr>
<tr><td align="left">torchxla+TPU</td>
<td align="center">85.9</td>
</tr>
</tbody></table>