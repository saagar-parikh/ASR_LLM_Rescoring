###############################################################################
## Description: Creates a reference dictionary storing the correct ASR       ##
##              reference sentence for each ASR utterance                    ##
## Written by: Justin Dannemiller                                            ##
## Last Modified: 04 December 2023                                           ##
###############################################################################


import os
import argparse
import json
from preprocess_data import extract_sentence, extract_uttID

""""
@brief: Creates a dictionary of references sentences with which the ASR 
       hypotheses should be compared to.
@param[in] path_to_refs - Path to the reference sentences
@return ref_dict: Dictionary mapping utterance ID to the corrresponding
                  transcribed sentence
"""


def create_reference_dict(path_to_refs):
    ## Read all lines in the reference file
    with open(path_to_refs, "r") as ref_file:
        reference_lines = ref_file.readlines()
    ref_dict = {}
    ## Add each reference sentence to reference dictionary
    for ref_line in reference_lines:
        uttID = extract_uttID(ref_line)
        reference = extract_sentence(ref_line)
        ref_dict[uttID] = reference
    return ref_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="test_other")
    args = parser.parse_args()

    current_path = os.getcwd()
    test_set = args.test_set
    reference_path = os.path.join(current_path, "ground_truth", test_set, "text")

    ref_dict = create_reference_dict(reference_path)
    ref_dict_file_name = "ref_dict_" + test_set + ".json"
    with open(ref_dict_file_name, "w") as ref_json_file:
        json.dump(ref_dict, ref_json_file, indent=4)
