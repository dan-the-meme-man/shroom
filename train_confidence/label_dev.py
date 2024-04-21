from huggingface_hub import hf_hub_download
from datasets import load_dataset
from json import dump, load
from os.path import join
from numpy import exp
from numpy.random import seed
from random import randint
from tqdm.notebook import trange
from llama_cpp import Llama

import random
seed_val = 442
random.seed(seed_val)
seed(seed_val)

run_on_test = False # whether this baseline system is ran on the test splits or the val splits

in_dir = 'SHROOM_dev-v2'
out_dir = 'solar_out'

# change paths appropriately
# make sure the output filename is the same as the reference filename for the scoring program
path_val_model_aware = join(in_dir, 'val.model-aware.v2.json')
path_val_model_aware_output = join(out_dir, 'val.model-aware.v2.json')

# change paths appropriately
# make sure the output filename is the same as the reference filename for the scoring program
path_val_model_agnostic = join(in_dir, 'val.model-agnostic.json')
path_val_model_agnostic_output = join(out_dir, 'val.model-agnostic.json')

# simple JSON loading
with open(path_val_model_aware, 'r') as istr:
    data_val_all = load(istr)
num_sample = len(data_val_all)
print(num_sample)

exit()

# This config has been tested on an RTX 3080 (VRAM of 16GB).
# you might need to tweak with respect to your hardware.

model_name_or_path = "upstage/SOLAR-10.7B-Instruct-v1.0"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=16, # CPU cores
    n_batch=8000, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32, # Change this value based on your model and your GPU VRAM pool.
    n_ctx=8192, # Context window
    logits_all=True
)

output_json = []
labels = ["Not Hallucination", "Hallucination"]
for i in trange(num_sample):
    # label = str(data_val_all[i]['label'])
    task = str(data_val_all[i]['task'])
    if run_on_test:
        # test splits will contain ids to ensure correct alignment before scoring
        id = int(data_val_all[i]['id'])
    hyp = str(data_val_all[i]['hyp'])
    src = str(data_val_all[i]['src'])
    tgt = str(data_val_all[i]['tgt'])

    if task == "PG":
        context = f"Context: {src}"
    else: #i.e. task == "MT" or task == "DM":
        context = f"Context: {tgt}"

    sentence = f"Sentence: {hyp}"
    message = f"{context}\n{sentence}\nIs the Sentence supported by the Context above? Answer using ONLY yes or no:"
    prompt = f"<s>[INST] {message} [/INST]"

    response = lcpp_llm(
        prompt=prompt,
        temperature= 0.0,
        logprobs=1,
    )
    answer = str(response["choices"][0]["text"]).strip().lower()
    if answer.startswith("yes"):
        output_label = "Not Hallucination"
        prob = 1-float(exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
    if answer.startswith("no"):
        output_label = "Hallucination"
        prob = float(exp(response["choices"][0]["logprobs"]["token_logprobs"][0]))
    if not answer.startswith("no") and not answer.startswith("yes"):
        idx_random = randint(0,len(labels)-1)
        output_label = labels[idx_random]
        prob = float(0.5)

    item_to_json = {"label":output_label, "p(Hallucination)":prob}
    if run_on_test:
        item_to_json['id'] = id
    output_json.append(item_to_json)

f = open(path_val_model_agnostic_output, 'w+', encoding='utf-8')
dump(output_json, f)
f.close()
print("done")