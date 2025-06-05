CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path ciCic/llama-3.2-1B-Instruct-AWQ --port 30000
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --model-path meta-llama/Llama-3.2-1B-Instruct --port 30002
CUDA_VISIBLE_DEVICES=2 python scripts/test_time_compute.py recipes/Llama-3.2-1B-Instruct/beam_search.yaml