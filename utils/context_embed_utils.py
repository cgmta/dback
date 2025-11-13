import os
import torch
import numpy as np
from functools import partial
from glob import glob

import faiss
from datasets import Features, Sequence, Value, concatenate_datasets, load_from_disk

from transformers import AutoTokenizer, AutoModel
from pyserini.encode import AnceEncoder

def embed_update_ance(hparams, world_size: int=1, rank: int=0, device: str='cuda'):    
    
    # Load the context encoder and tokenizer
    # Load the context dataset
    # map the context dataset to embeddings
    # for ance type model 
    
    context_tokenizer = AutoTokenizer.from_pretrained(hparams.ctx_tokenizer)
    ctx_encoder = AnceEncoder.from_pretrained(hparams.ctx_encoder).to(device)
    context_dataset = load_from_disk(hparams.wiki_noemb)
    
    context_list = [context_dataset.shard(world_size, i, contiguous=True) for i in range(world_size)]
    data_shard = context_list[rank]

    arrow_folder = "data_" + str(rank)
    embed_path = os.path.join(hparams.base_dir, "wiki/shard_dir", arrow_folder)
    os.makedirs(embed_path, exist_ok=True)
    print(f"Rank {rank} is processing {embed_path}")

    def embed(
        documents: dict, ctx_encoder: AutoModel, ctx_tokenizer: AutoTokenizer, device
    ) -> dict:
        """Compute the DPR embeddings of document passages"""
        # texts = [f'{title} {text}' for title, text in zip(documents["title"], documents["text"])]
        texts = documents["text"]
        input_ids = ctx_tokenizer(
            texts,    # documents["title"], 
            max_length=256,
            padding='longest',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        # input_ids = {key: value.to(device) for key, value in input_ids.items()}
        input_ids.to(device)
        with torch.no_grad():
            embeddings = ctx_encoder(input_ids["input_ids"]).detach().cpu().numpy()
        return {"embeddings": embeddings}

    new_features = Features(
        {"id": Value("string"), "text": Value("string"),       # 注意 id key
         "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space

    dataset = data_shard.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=context_tokenizer, device=device),
        batched=True,
        batch_size=32,  # Adjust batch size as needed
        features=new_features,
    )
    dataset.save_to_disk(embed_path)

def embed_update(hparams, world_size: int=1, rank: int=0, device: str='cuda'):    
    
    # Load the context encoder and tokenizer
    # Load the context dataset
    # map the context dataset to embeddings
    # for dpr type model 
    
    context_tokenizer = AutoTokenizer.from_pretrained(hparams.ctx_tokenizer)
    ctx_encoder = AutoModel.from_pretrained(hparams.ctx_encoder).to(device)
    context_dataset = load_from_disk(hparams.wiki_noemb)
    
    context_list = [context_dataset.shard(world_size, i, contiguous=True) for i in range(world_size)]
    data_shard = context_list[rank]

    arrow_folder = "data_" + str(rank)
    embed_path = os.path.join(hparams.base_dir, "wiki/shard_dir", arrow_folder)
    os.makedirs(embed_path, exist_ok=True)
    print(f"Rank {rank} is processing {embed_path}")

    def embed(
        documents: dict, ctx_encoder: AutoModel, ctx_tokenizer: AutoTokenizer, device
    ) -> dict:
        """Compute the DPR embeddings of document passages"""
        input_ids = ctx_tokenizer(
            documents["text"],    # documents["title"], 
            truncation=True, max_length=512, padding=True, return_tensors="pt"
        )
        input_ids = {key: value.to(device) for key, value in input_ids.items()}
        with torch.no_grad():
            embeddings = ctx_encoder(**input_ids, return_dict=True).pooler_output
        return {"embeddings": embeddings.detach().cpu().numpy()}

    new_features = Features(
        {"id": Value("string"), "text": Value("string"),       # 注意 id key
         "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space

    dataset = data_shard.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=context_tokenizer, device=device),
        batched=True,
        features=new_features,
    )
    dataset.save_to_disk(embed_path)
    
def cor_embed_update_muti_process(model, context_dataset, hparams, world_size, rank, device):
    cor_model = model.to(device)
    # context_dataset = load_from_disk(hparams.passages_path)
    context_list = [context_dataset.shard(world_size, i, contiguous=True) for i in range(world_size)]
    data_shrad = context_list[rank]
    
    def embed(examples, model, device):
        embeddings = torch.tensor(examples['embeddings']).to(device)
        with torch.no_grad():
            cor_embedding = model(embeddings)
        return {"embeddings": cor_embedding.detach().cpu().numpy()}
    
    dataset = data_shrad.map(
        partial(embed, model=cor_model, device=device),
        batched=True,
    )
    dataset.save_to_disk(hparams.shard_dir)
    
def cor_embed_update_one_process(model, hparams, device, step_count):
    if step_count == hparams.indexing_freq:
        context_dataset = load_from_disk(hparams.passages_path_base)
    else:
        context_dataset = load_from_disk(hparams.passages_path)
    
    cor_model = model
    
    def embed(examples, model, device):
        embeddings = torch.tensor(examples['embeddings']).to(device)
        with torch.no_grad():
            cor_embedding = model(embeddings)
        return {"embeddings": cor_embedding.detach().cpu().numpy()}
    
    context_dataset = context_dataset.map(
        partial(embed, model=cor_model, device=device),
        batched=True,
        num_proc=2,
    )
    context_dataset.save_to_disk(hparams.passages_path)
    del context_dataset

def concat_dataset(shard_dir):
    data_shard_list = []
    for shard_address in glob(str(shard_dir) + "/*/"):
        data_shard_list.append(load_from_disk(shard_address))
    concat = concatenate_datasets(data_shard_list)
    return concat

def add_hnsw_flat_index(dataset, index_path):
    faiss.omp_set_num_threads(96)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    dataset.get_index("embeddings").save(
        index_path
    )  # since we load the index in to memory,we can directly update the index in the disk
    return dataset
    
def add_compress_index(dataset, index_path):
    d = 768
    train_size=262144 

    quantizer = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = 200
    quantizer.hnsw.efSearch = 128
    ivf_index = faiss.IndexIVFPQ(quantizer, d, 4096, 128, 8, faiss.METRIC_INNER_PRODUCT)
    ivf_index.nprobe = 64
    ivf_index.own_fields = True
    quantizer.this.disown()

    sample = np.array(dataset["embeddings"][:train_size])
    if sample.dtype != np.float32:
        sample = sample.astype(np.float32)
    faiss.normalize_L2(sample) 
    ivf_index.train(sample)

    dataset.add_faiss_index(
    "embeddings",
    train_size=train_size,
    custom_index=ivf_index,
    )
    print("Saving wiki_dpr compress faiss index")
    
    dataset.save_faiss_index("embeddings", index_path) 
    return dataset   
    
def add_flat_index(dataset, index_path):
    dataset.add_faiss_index(column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT)
    dataset.get_index("embeddings").save(
        index_path
    ) 
    return dataset


if __name__ == "__main__":
    # Path to the large .npy file containing the vectors
    file_path = "/data/coign/tc/project/studyrpogram/retriver/Base_Model_and_Data/dataset/wikidir_dpr_mutiset_base_line/wiki_noindex"
    dataset = load_from_disk(file_path)
    # Path to save the FAISS index
    save_path = "/data/coign/tc/project/studyrpogram/retriver/Base_Model_and_Data/dataset/wikidir_dpr_mutiset_base_line/index/falt_index.faiss"
    
    # Construct the FAISS index
    add_flat_index(dataset, save_path)