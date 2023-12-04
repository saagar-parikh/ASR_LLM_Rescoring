###############################################################################
## Description: Preprocesses the numerous files that store the ASR           ##
##              hypotheses and their corresponding scores such that the      ##
##              n-best hypotheses and their scores can be directly accessed  ##
##              and processed downstream                                     ##
## Written by: Justin Dannemiller                                            ##
## Modified by: Saagar Parikh                                                ##
## Last Modified: 02 December 2023                                           ##
###############################################################################

import os
import pathlib
import json
import argparse

""" 
brief: Extracts the utterance ID from a line from hypothesis file
param[in]: line_from_file: line taking from hypothesis file that contains
           the utterance ID followed by the corresponding hypothesis
returns: utt_ID - The utterance ID for the corresponding hypothesis
"""
def extract_uttID(line_from_file):
    utt_ID = line_from_file.split(" ")[0]
    return utt_ID
""" 
brief: Extracts the hypothesis form a line from hypothesis file
param[in]: line_from_file - line taking from hypothesis file that contains
           the utterance ID followed by the corresponding hypothesis
returns: hypothesis - The extracted hypothesis
"""
def extract_hypothesis(line_from_file):
    hyp_as_list = line_from_file.split(" ")[1:]
    # Convert from list to str
    hypothesis = " ".join(hyp_as_list)
    return hypothesis
""" 
brief: Extracts the score assigned to a given ASR hypothesis from a line in
       a score file
param[in]: line_from_file - line taking from score file that contains
           the utterance ID followed by the score assigned to the corresponding
           hypothesis
returns: score - The extracted score
"""
def extract_score(line_from_file):
    tensor_score_str = line_from_file.split(" ")[-1]
    score = tensor_score_str.split("(")[-1].split(")")[0]
    return float(score)

"""
@brief Inserts all the hypotheses and scores from a given kth_best directory
       into the provided hypothesis dictionary
param[out] hyp_dict - The hypothesis dictionary into which to insert
param[in]  path_to_ibest - Path to the ibest directory storing the scores and 
           hypotheses
returns: hyp_dict - The hypothesis dictionary with newly inserted hyps and 
                    scores
"""
def update_hyp_dict(hyp_dict, path_to_ibest):
    path_to_hyps = os.path.join(path_to_ibest, "text")
    path_to_scores = os.path.join(path_to_ibest, "score")

    # Open the two files and read their lines
    with open(path_to_hyps, 'r') as hyp_file:
        hyp_file_lines = hyp_file.readlines()
    with open(path_to_scores, 'r') as score_file:
        score_file_lines = score_file.readlines()

    for hyp_line in hyp_file_lines:
        utt_ID = extract_uttID(hyp_line)
        hypothesis = extract_hypothesis(hyp_line)
        if utt_ID not in hyp_dict:
            hyp_dict[utt_ID] = {"hypotheses": [], 
                                "scores": []}
        hyp_dict[utt_ID]["hypotheses"].append(hypothesis)
    
    for score_line in score_file_lines:
        utt_ID = extract_uttID(score_line)
        score = extract_score(score_line)
        if utt_ID not in hyp_dict:
            hyp_dict[utt_ID] = {"hypotheses": [], 
                                "scores": []}
        hyp_dict[utt_ID]["scores"].append(score)
    return hyp_dict


""" 
brief: Creates a dictionary that maps an utterance to a set of n-best 
       and the corresponding scores output from Ebranchformer model 
param[in] data_dir_path - Path of the folder storing all of the output of

returns utt_to_hyp - Dictionary mapping from utterance ID to hypotheses and 
                     scores
"""
def create_hypothesis_dict(data_path):
    # Return error if given data path is not one of the four acceptable sets
    permissible_dirs = ["dev_clean", "dev_other", "test_clean", "test_other"]
    data_path = pathlib.PurePath(data_path)
    if data_path.name not in permissible_dirs:
        print("Error: Invalid directory! Must be the path of one of the four folders\n"
              "1) dev_clean\n2) dev_other\n3) test_clean\n4) test_other")
        exit(1)

    else:
        hypothesis_dict = {}
        # Create list of paths to the 8 output.x directories output from the 
        # 8 CPU units used to run ninference
        path_to_logdir = os.path.join(data_path, "logdir")
        logdir_contents = os.listdir(path_to_logdir)
        output_dir_names = [dir_name for dir_name in logdir_contents if "output." in dir_name]
        for output_dir in output_dir_names:
            output_path = os.path.join(path_to_logdir, output_dir)
            # Get a list of the nbest directories from which to extract the hyps and
            # scores
            output_contents = os.listdir(output_path)
            nbest_dir_names = [dir_name for dir_name in output_contents if "best_recog" in dir_name]
            for ibest_dir in nbest_dir_names:
                ibest_path = os.path.join(output_path, ibest_dir)
                hypothesis_dict = update_hyp_dict(hypothesis_dict, ibest_path)
        return hypothesis_dict
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default='test_other')
    args = parser.parse_args()

    current_path = os.getcwd()
    test_set = args.test_set
    test_other_path = os.path.join(current_path, "inference", test_set)
    hyp_dict1 = create_hypothesis_dict(test_other_path)

    with open("hyp_dict_"+test_set+".json", "w") as outfile:
        json.dump(hyp_dict1, outfile, indent=4)

    with open("hyp_dict_"+test_set+".json", 'r') as f:
        hyp_dict = json.load(f)

    print(f"Length of dictionary: {len(hyp_dict)}")

    # Demonstate structure of the dictionary using a given utterance ID from
    # the test_other dataset
    # hyps_and_scores = hyp_dict["2609-156975-0007"]
    # hyps = hyps_and_scores["hypotheses"]
    # scores = hyps_and_scores["scores"]
    # for hyp,score in zip(hyps,scores):
    #     print("hypothesis: " + hyp + "\tscore:" + str(score))

