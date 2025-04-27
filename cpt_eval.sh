export CUDA_VISIBLE_DEVICES=3 
export PYTHONPATH=. 

# offline=True

# python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=midupper3-mlp" --offline ${offline}

# python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=all" --offline ${offline}

offline=False

# python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=midupper3-mlp" 

python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=all" 

