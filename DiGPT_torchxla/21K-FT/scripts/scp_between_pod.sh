export PROJECT_ID=proj_name
export ZONE=zone
export TPU_NAME=tpu_name

gcloud compute config-ssh # need to configure ssh first time
gcloud alpha compute tpus tpu-vm scp --recurse /path/to/21K-FT/  $TPU_NAME:/path/to/ --zone=$ZONE --worker=all
