# ASR LLM Rescoring

## Instructions

1. Run `preprocess_data.py` to generate dictionaries containing n-best asr scores for each utterance.
2. Run `lllm_scoring.py` to update dictionaries with llm scores for each utterance. (for `gpt2` and `bert`)
3. Run `combined_scores.py` with arg `--lambda_param` to combine the asr and llm scores. 
4. Run `compute_error_rate.py` to compute the error rate for a given hypothesis dictionary.
5. `gridsearch.sh` Tests error rates on a range of lambda values.
6. `hyp_comb_10_dict_test_other.json` contains the hypotheses and all the scores for the automasking experiment
7. `hyp_comb_masks_10_dict_test_other.json` contains the hypotheses and all the scores for the selective mask-based experiment

