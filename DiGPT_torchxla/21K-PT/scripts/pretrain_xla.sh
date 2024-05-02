export PROJECT_ID=proj_name
export ZONE=zone
export TPU_NAME=tpu_name
export XRT_TPU_CONFIG='localservice;0;localhost:51011'

# run this script on a TPU v3-256 machine
python3 -m torch_xla.distributed.xla_dist \
--tpu=${TPU_NAME} \
--restart-tpuvm-pod-server \
--env TORCH_CUDNN_V8_API_ENABLED=1 \
--env TFDS_PREFETCH_SIZE=256 \
--env TFDS_SHUFFLE_SIZE=4096 \
-- python3 /path/to/workdir/launch_xla.py --num-devices 8 \
main_pretrain.py \
--batch_size 32 \
--model DiGPT_vit_huge \
--name 21k \
--output_dir /path/to/workdir/output_dir \
--clip_path /path/to/workdir/openclip_vit_h_14.pth \
--epochs 150 --warmup_epochs 5 \
--blr 1e-4 --weight_decay 0.05 \
--data_path gs://path/to/in21k_tfrecord/ --num_workers 3 \
--src /path/to/workdir/output_dir/21k/ \
--des gs://path/to/cloud/output_dir/ \
2>&1 | tee /path/to/log.txt