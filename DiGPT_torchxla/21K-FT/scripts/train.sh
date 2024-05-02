export PROJECT_ID=proj_name
export ZONE=zone
export TPU_NAME=tpu_name
export XRT_TPU_CONFIG='localservice;0;localhost:51011'


python3 -m torch_xla.distributed.xla_dist \
--tpu=${TPU_NAME} \
--restart-tpuvm-pod-server \
--env TORCH_CUDNN_V8_API_ENABLED=1 \
--env TFDS_PREFETCH_SIZE=64 \
--env TFDS_SHUFFLE_SIZE=2048 \
-- python3 /path/to/21K-FT/launch_xla.py --num-devices 8 \
main_finetune.py \
    --batch_size 8 --input_size 448 --finetune /path/to/21K-FT/pretrain_ckpt.pth  \
    --model vit_huge_patch14 --nb_classes 19167 \
    --epochs 30 --warmup_epochs 1 --name 21k \
    --output_dir /path/to/21K-FT/out_finetune_448/ \
    --blr 3.75e-5 --layer_decay 0.75 --num_workers 3 \
    --weight_decay 0.05 --drop_path 0.3 --reprob 0. --mixup 0. --cutmix 0. \
    --data_path gs://path/to/21k_tfrecord/ \
    --src /path/to/21K-FT/out_finetune_448/21k/ \
    --des gs://path/to/21K-FT/output_dir/ \
2>&1 | tee /path/to/21K-FT/log.txt


# lr=4e-4 bz 2048
