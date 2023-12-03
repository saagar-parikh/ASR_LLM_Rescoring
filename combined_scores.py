import torch
import json
import argparse
from torch.nn import functional as F


def rerank(dict_, id, key):
    # Get the index of the sort key in the arrays
    sort_index = list(dict_[id].keys()).index(key)

    # Sort the arrays based on the sort key
    sorted_values = [
        list(t)
        for t in zip(
            *sorted(
                zip(*[dict_[id][key] for key in dict_[id]]), key=lambda x: x[sort_index]
            )
        )
    ]

    # Update the dictionary with the sorted values
    for key, values in zip(dict_[id].keys(), sorted_values):
        dict_[id][key] = values

    return dict_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="test_other")
    parser.add_argument("--nbest", type=int, default=5)
    parser.add_argument("--lambda_param", type=int, default=0.3)
    args = parser.parse_args()

    test_set = args.test_set
    nbest = args.nbest
    lambda_param = args.lamda_param

    with open("hyp_llm_10_dict_" + test_set + ".json", "r") as f:
        hyp_dict = json.load(f)
    final_dict_gpt2 = {}
    final_dict_bert = {}
    gpt2_dict = {}
    bert_dict = {}
    for utt_id in hyp_dict:
        sentences = hyp_dict[utt_id]["hypotheses"]
        asr_scores = hyp_dict[utt_id]["scores"]
        gpt2_scores = hyp_dict[utt_id]["gpt2_scores"]
        bert_scores = hyp_dict[utt_id]["bert_scores"]
        ## if want to take nbest sorting will be required
        asr_scores_softmax = F.softmax(torch.abs(torch.tensor(asr_scores))).tolist()
        hyp_dict[utt_id]["final_score_gpt2"] = (
            1 - lambda_param
        ) * asr_scores_softmax + lambda_param * gpt2_scores
        hyp_dict[utt_id]["final_score_bert"] = (
            1 - lambda_param
        ) * asr_scores_softmax + lambda_param * bert_scores

        # Key to sort by

        gpt2_dict[utt_id]["hypotheses"] = hyp_dict[utt_id]["hypotheses"]
        gpt2_dict[utt_id]["final_score_gpt2"] = hyp_dict[utt_id]["final_score_gpt2"]

        bert_dict[utt_id]["hypotheses"] = hyp_dict[utt_id]["hypotheses"]
        bert_dict[utt_id]["final_score_bert"] = hyp_dict[utt_id]["final_score_bert"]

        gpt2_dict = rerank(gpt2_dict, utt_id, "final_score_gpt2")
        bert_dict = rerank(bert_dict, utt_id, "final_score_bert")
        # Print the updated dictionary

        print(
            "Sentence: {} and Combined Score for gpt2: {}".format(
                gpt2_dict[utt_id]["hypotheses"][0],
                gpt2_dict[utt_id]["final_score_gpt2"][0],
            )
        )
        print(
            "Sentence: {} and Combined Score for bert: {}".format(
                gpt2_dict[utt_id]["hypotheses"][0],
                gpt2_dict[utt_id]["final_score_bert"][0],
            )
        )

        final_dict_gpt2[utt_id] = gpt2_dict[utt_id]["hypotheses"][0]
        final_dict_bert[utt_id] = gpt2_dict[utt_id]["hypotheses"][0]

    with open("hyp_comb_" + str(nbest) + "_dict_" + test_set + ".json", "w") as outfile:
        json.dump(hyp_dict, outfile, indent=2)
    with open(
        "final_hyp_gpt2_" + str(nbest) + "_dict_" + test_set + ".json", "w"
    ) as outfile1:
        json.dump(final_dict_gpt2, outfile1, indent=2)
    with open(
        "final_hyp_bert_" + str(nbest) + "_dict_" + test_set + ".json", "w"
    ) as outfile2:
        json.dump(final_dict_bert, outfile2, indent=2)
    with open(
        "rerank_hyp_gpt2_" + str(nbest) + "_dict_" + test_set + ".json", "w"
    ) as outfile3:
        json.dump(gpt2_dict, outfile3, indent=2)
    with open(
        "rerank_hyp_bert_" + str(nbest) + "_dict_" + test_set + ".json", "w"
    ) as outfile4:
        json.dump(bert_dict, outfile4, indent=2)


if __name__ == "__main__":
    main()
