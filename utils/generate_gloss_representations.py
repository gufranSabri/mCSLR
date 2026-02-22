
import os
import torch
from tqdm import tqdm
import pickle
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



llm = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large", cache_dir="./data/models").to("cuda")
tokenizer = AutoTokenizer.from_pretrained(
    "bigscience/mt0-large", 
    cache_dir="./data/models",
    max_length=64,
)


data_dir = "/home/ahmedubc/projects/aip-lsigal/ahmedubc/datasets"
datasets = ["CSL-Daily", "phoenix2014-T"]



for dataset in datasets:
    for mode in ['train', 'dev', 'test']:
        pickle_path = os.path.join(data_dir, dataset, f"pose.{mode}")

        emb_data = {}

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        for key in tqdm(data):
            gls_ref = [data[key]['gloss']]
            inputs = tokenizer(gls_ref, return_tensors="pt", truncation=True, max_length=64)

            output_tokens = tokenizer(
                gls_ref,
                padding="longest",
                return_tensors="pt",
            ).to("cuda")
            
            text_embeds = llm.encoder.embed_tokens(output_tokens.input_ids)
            emb_data[key] = text_embeds.cpu().detach()
    
        # save 
        save_path = os.path.join(data_dir, dataset, f"gloss_{dataset}.{mode}")
        with open(save_path, 'wb') as f:
            pickle.dump(emb_data, f)
        
        print(f"Saved gloss embeddings for {dataset} {mode} at {save_path}")
            