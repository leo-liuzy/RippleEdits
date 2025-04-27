export CUDA_VISIBLE_DEVICES=3 
export PYTHONPATH=. 

# offline=True


python src/evaluation_leo.py --editor memit --exp llama3.2-1B-eos-sft-mid-upper  --offline
# python src/evaluation_leo.py --editor memit --exp llama3.2-1B-eos-sft-mid-upper 
# python src/evaluation_leo.py --editor memit --exp llama3.2-1B-eos-sft-top  --offline
# python src/evaluation_leo.py --editor memit --exp llama3.2-1B-eos-sft-top 