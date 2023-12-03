# ASR LLM Rescoring

## Instructions

1. Run `preprocess_data.py` to generate dictionaries containing n-best asr scores for each utterance.
2. Run `lllm_scoring.py` to update dictionaries with llm scores for each utterance. (for `gpt2` and `bert`)
3. Run `combined_scores.py` with arg `--lambda_param` to combine the asr and llm scores. 
