###############################################################################
## Description: Computes the word error rate (WER) and character error rate  ##
##              (CER) of the best ASR hypotheses as measured by the combined ##
##              scores given by the ASR-LLM rescoring system.                ##
## Written by: Justin Dannemiller                                            ##
## Last Modified: 03 December 2023                                           ##
###############################################################################

import os
import json
import argparse
import jiwer
import numpy as np


"""
@brief: Extracts the most probable rescored hypothesis correponding to a 
       particular utterance
@param[in] rescored_hyp_dict: A dictionary containing all the ASR hypotheses,
                              ASR-LLM rescored scores, etc. *Note, this is not
                              the same as hyp_dict. It is hyp_dict[utt_id]

@param[in] score_metric: A string describing which set of scores from the 
                          dictionary should be used when extracting the max
                          probability hypothesis. Must be "asr_scores_softmax",
                          "final_score_gpt2", or "final_score_bert"
return: max_prob_hyp: The most probable hypothesis corresponding to a 
                      particular utterance"""
def extract_max_prob_hyp(rescored_hyp_dict, score_metric):
    # Get the maximum probability hypothesis
    LLM_rescored_values = rescored_hyp_dict[score_metric]
    max_prob_idx = np.argmax(LLM_rescored_values)
    max_prob_hyp = rescored_hyp_dict["hypotheses"][max_prob_idx]
    return max_prob_hyp

"""
@brief: Creates a list of the most probable rescored hypotheses across all 
        utterances
@param[in] hyp_dict: A dictionary mapping any provided utterance ID to 
                     corresponding lists of ASR hypotheses, ASR scores,
                     combined scores, etc.
@param[in] score_metric: A string describing which set of scores from the 
                          dictionary should be used when extracting the max
                          probability hypothesis. Must be "asr_scores_softmax",
                          "final_score_gpt2", or "final_score_bert"
@return best_hypotheses: The list of most probable hypotheses across 
                              all utterances
"""
def get_best_hypotheses(hyp_dict, score_metric):
    # Check that the given LLM model selected is within the permissible options
    permissible_metrics = ["asr_scores_softmax", "final_score_gpt2", "final_score_bert"]
    if score_metric not in permissible_metrics:
        print(f"Provided LLM model paramater: {score_metric} not in the set of \
                permitted options.\nPermitted options include gpt2 and bert!")
    else:
        best_hypotheses = []
        uttIDs = hyp_dict.keys()
        # Collect the best hypotheses for each utterance
        for uttID in uttIDs:
            hyps_for_utt_dict = hyp_dict[uttID]
            max_prob_hyp = extract_max_prob_hyp(hyps_for_utt_dict, score_metric)
            best_hypotheses.append(max_prob_hyp)    
    return best_hypotheses

# """
# @brief: Computes the average word error rate between a given list of hypotheses
#         and the corresponding list of reference sentences
# @param[in] hypotheses: List of ASR-transcribed hypotheses
# @param[in] reference_sent: List of corresponding sentences
# """
# def compute_WER(hypotheses, reference_sent):
#     # Compute WER for each sentence
#     word_err_rates = [jiwer.wer()]

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="test_other")
    args = parser.parse_args()
    test_set = args.test_set

    ### Get the reference transcription sentences
    ref_dict_file_name = "ref_dict_" + test_set + ".json"
    ref_dict_path = os.path.join(os.getcwd(), ref_dict_file_name) 
    with open(ref_dict_path, 'r') as ref_dict_file:
        ref_dict = json.load(ref_dict_file)
    # Sort the dictionary to ensure uniform ordering between reference and hyps
    ref_dict=dict(sorted(ref_dict.items()))
    ref_sentences = list(ref_dict.values())

    ### Get the best hypotheses
    rescored_hyp_file_name = "hyp_comb_10_dict_"+ test_set + ".json"
    rescored_dict_path = os.path.join(os.getcwd(), rescored_hyp_file_name)
    with open(rescored_dict_path, 'r') as rescored_dict_file:
        rescored_dict = json.load(rescored_dict_file)
    # Sort the dictionary to ensure uniform ordering between reference and hyps
    rescored_dict = dict(sorted(rescored_dict.items()))

    gpt_score_metric= "final_score_gpt2"
    gpt_best_hyps = get_best_hypotheses(rescored_dict, gpt_score_metric)
    bert_score_metric = "final_score_bert"
    bert_best_hyps = get_best_hypotheses(rescored_dict, bert_score_metric)
    asr_score_metric = "asr_scores_softmax"
    asr_best_hyps = get_best_hypotheses(rescored_dict, asr_score_metric)

    ### Compute word error rate
    gpt2_wer= jiwer.wer(ref_sentences, gpt_best_hyps)
    bert_wer = jiwer.wer(ref_sentences, bert_best_hyps)
    asr_wer = jiwer.wer(ref_sentences, asr_best_hyps)

    score_names = ["gpt2", "bert", "asr"]
    WER_values = [gpt2_wer, bert_wer, asr_wer]
    for i in range(len(score_names)):
        print(f"The {score_names[i]} word error rate is: {WER_values[i]}")
