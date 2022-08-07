from model_wrapper import ModelForMaskedLM
import torch
import torch.nn as nn
import os
import json
import pickle
import numpy as np
import argparse
import utils
from transformers import RobertaTokenizerFast
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
args = parser.parse_args()

with open(os.path.join(args.model, 'config.json'), 'r') as f:
    config = json.load(f)

model_config = config["model"]
pretraining_config = config["pretraining_setting"]
gpu_config = config["gpu_setting"]
checkpoint_dir = os.path.join(args.model, 'model')
dataset = config["dataset"]

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, pretraining_config], indent = 4))

########################### Loading Model ###########################

model = ModelForMaskedLM(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

if "from_cp" in config:

    from_cp = config["from_cp"]
    checkpoint = torch.load(from_cp, map_location = 'cpu')

    cp_pos_encoding = checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'].data.numpy()
    cp_max_seq_len, embedding_dim = cp_pos_encoding.shape
    assert model_config["max_seq_len"] >= (cp_max_seq_len - 2)
    assert model_config["max_seq_len"] % (cp_max_seq_len - 2) == 0
    num_copy = model_config["max_seq_len"] // (cp_max_seq_len - 2)
    pos_encoding = np.concatenate([cp_pos_encoding[:2, :]] + [cp_pos_encoding[2:, :]] * num_copy, axis = 0)
    checkpoint['model_state_dict']['model.embeddings.position_embeddings.weight'] = torch.tensor(pos_encoding, dtype = torch.float)

    utils.load_model_ignore_mismatch(model.module, checkpoint['model_state_dict'])
    print("Model initialized", from_cp, flush = True)
    
    dump_path = os.path.join(checkpoint_dir, f"init.model")
    torch.save({
        "model_state_dict":model.module.state_dict()
    }, dump_path)

elif config["from_pretrained_roberta"]:

    roberta_path = os.path.join("roberta-base-pretrained.pickle")
    with open(roberta_path, "rb") as f:
        weights = pickle.load(f)

    assert weights["model.embeddings.position_embeddings.weight"].shape[0] == (model_config["max_seq_len"] + 2)

    for key in weights:
        weights[key] = torch.tensor(weights[key])

    utils.load_model_ignore_mismatch(model.module, weights)
    print("Model initialized from RoBERTa", flush = True)

else:
    print("Model randomly initialized", flush = True)

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
text = "The <mask> of Belgium is Brussels."

inputs = tokenizer(text, return_tensors="pt", padding = 'max_length')

with torch.no_grad():
    logits = model(inputs.input_ids)

print('logits', logits, logits.size())

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
print(mask_token_index)
mask_token_logits = logits[0, mask_token_index, :].detach().cpu()

print(mask_token_logits.size())

top_5_tokens = np.argsort(-mask_token_logits)[:5].tolist()
print(top_5_tokens)

#for token in top_5_tokens:
#    print(f">>> {text.replace(tokenizer.mask_token, tokenizer.decode(token))}")