export PROJECT_ID=proj_name
export ZONE=zone
export TPU_NAME=tpu_name

## prepara env && login wandb
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command \
" cd /path/to/workdir/scripts/ && bash prepare_env_pods.sh"


for num in {0..31}
do
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=$num --command "sudo pkill -f python3"
done
