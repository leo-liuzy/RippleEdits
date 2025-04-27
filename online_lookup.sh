export CUDA_VISIBLE_DEVICES=3 
export PYTHONPATH=. 

offline=False

python src/evaluation_leo.py --editor mend --exp "ripple_edits_all_original_mend_noshare_midupper3"

python src/evaluation_leo.py --editor mend --exp "ripple_edits_all_original_mend_share_top3"

python src/evaluation_leo.py --editor mend --exp "zereFull_original_mend_share_top3"

python src/evaluation_leo.py --editor mend --exp "zereFull_original_mend_noshare_midupper3" 

# python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=midupper3-mlp" --offline ${offline}

# python src/evaluation_leo.py --editor cpt --exp "Llama-3.2-1B-eos-sft_clm-baseline_lr=1e-05_epoch=4.0_tuned-params=all" --offline ${offline}


