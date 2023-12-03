import torch
import json
import argparse
from tqdm import tqdm
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM


def score(sentence, tokenizer, model, device):
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    tokens = tokens.to(device)
    with torch.no_grad():
        outputs = model(tokens, labels=tokens)
        loss = outputs.loss
        probability = torch.exp(-loss)

    return probability.item()


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are {} GPUs available.".format(torch.cuda.device_count()))
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="test_other")
    parser.add_argument("--nbest", type=int, default=5)
    args = parser.parse_args()

    test_set = args.test_set
    nbest = args.nbest

    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
    model_bert = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    # tokenizer_gpt2.to(device)
    # tokenizer_bert.to(device)
    model_gpt2.to(device)
    model_bert.to(device)

    with open("hyp_dict_" + test_set + ".json", "r") as f:
        hyp_dict = json.load(f)

    for utt_id in hyp_dict:
        sentences = hyp_dict[utt_id]["hypotheses"]
        gpt2_scores = []
        bert_scores = []
        for sentence in tqdm(sentences[:nbest]):
            # sentence = torch.tensor(sentence).to(device)
            gpt2_scores.append(score(sentence, tokenizer_gpt2, model_gpt2, device))
            bert_scores.append(score(sentence, tokenizer_bert, model_bert, device))
        gpt2_scores = F.softmax(torch.tensor(gpt2_scores), dim=0).tolist()
        bert_scores = F.softmax(torch.tensor(bert_scores), dim=0).tolist()
        hyp_dict[utt_id]["gpt2_scores"] = gpt2_scores
        hyp_dict[utt_id]["bert_scores"] = bert_scores
        break

    with open("hyp_llm_" + str(nbest) + "dict_" + test_set + ".json", "w") as outfile:
        json.dump(hyp_dict, outfile, indent=2)


if __name__ == "__main__":
    main()
