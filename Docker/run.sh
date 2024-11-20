docker run -td \
  --name coigan_IROS2024_gpu_1 \
  --gpus '"device=1"'\
  --shm-size=150g \
  -v $(cat experiments_path.txt):/coigan/COIGAN-IROS-2024/experiments \
  -v $(cat datasets_path.txt):/coigan/COIGAN-IROS-2024/datasets \
  coigan-iros-2024:latest