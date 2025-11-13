"""
Define the retriever for the application
"""

import os
import ray
import faiss
import random
import datasets
import torch
import time
import numpy as np

import psutil

class Retriever:
    def __init__(self, config):
        self.config = config
        self.index = faiss.read_index(config.passage_index)
        self.dataset = datasets.load_from_disk(config.passage)
        self.embeddings = np.load(config.passage_embed)

        ## For testing purpose only ---
        # self.dataset = None
        # embeddings = np.random.rand(11_000_000, 768).astype(np.float32)
        # self.embeddings = np.concatenate([embeddings, embeddings], axis=0)
        # embeddings = None

    def get_dataset(self):
        return self.dataset
    
    def save_index(self, save_path):
        index_path = os.path.join(save_path, "index.faiss")
        faiss.write_index(self.index, index_path)
    
    def save_embeddings(self, save_path):
        embed_path = os.path.join(save_path, "embeddings.npy")
        np.save(embed_path, self.embeddings)
    
    def update_emb(self, ds_ids, updated_embeds):
        self.embeddings[ds_ids] = updated_embeds

    def update_embeddings(self, cor_model, batch_size=1024):
        num_embeddings = self.embeddings.shape[0]
        device = next(cor_model.parameters()).device
        emb_dtype = torch.from_numpy(self.embeddings[:1]).dtype
        for i, start_idx in enumerate(range(0, num_embeddings, batch_size)):
            end_idx = min(start_idx + batch_size, num_embeddings)
            batch_embeds = self.embeddings[start_idx:end_idx]
            batch_embeds_tensor = torch.tensor(np.copy(batch_embeds), dtype=emb_dtype, device=device)
            with torch.no_grad():
                updated_embeds_tensor = cor_model(batch_embeds_tensor)
            updated_embeds = (
                updated_embeds_tensor.detach().to("cpu")
                .numpy().astype(self.embeddings.dtype, copy=False)
            )
            self.embeddings[start_idx:end_idx] = updated_embeds
        print("Embeddings updated.")

    def update_flat_faiss(self):
        d = self.config.n_embd
        flat_index = faiss.IndexFlatIP(d)
        flat_index.add(self.embeddings)
        self.index = flat_index
        flat_index = None
        print("Flat FAISS index updated.")

    def update_qhnsw_faiss(self):
        d = self.config.n_embd
        quantizer = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
        quantizer.hnsw.efConstruction = 200
        quantizer.hnsw.efSearch = 128

        ivf_index = faiss.IndexIVFPQ(quantizer, d, 4096, 128, 8, faiss.METRIC_INNER_PRODUCT)
        ivf_index.nprobe = 128
        ivf_index.own_fields = True
        quantizer.this.disown()
        train_size = 262144
        ivf_index.train(self.embeddings[:train_size]) 
        ivf_index.add(self.embeddings)

        self.index = ivf_index    
        ivf_index = None 
        print("QHNSW FAISS index updated.")

    
    def retrieve(self, question_hidden_states, positive_ids):
        # For each question, retrieve self.config.n_docs candidates, then merge, deduplicate,
        # and insert positive samples. The positions of negatives and positives are randomized.
        # Example ordering: [neg, neg, ..., pos, neg, ...]. Negatives are randomly sampled;
        # positives are randomly inserted.
        _, all_ds_ids = self.index.search(question_hidden_states, self.config.n_docs + 10)
        ds_ids = list(all_ds_ids.flatten().tolist())

        # Convert positive_ids to integers and decrease by 1
        if self.config.model_type == "ance":
            positive_ids = [int(p_id) for p_id in positive_ids]
        else:
            positive_ids = [int(p_id) - 1 for p_id in positive_ids]
        ds_ids = [doc_id for doc_id in ds_ids if doc_id not in positive_ids]

        # Calculate target length for ds_ids after inserting positive_ids
        target_len = (
            len(positive_ids) * (self.config.n_docs - 1)
            + len(positive_ids)
            - len(set(positive_ids))
        )
        if len(ds_ids) > target_len:
            ds_ids = random.sample(ds_ids, target_len)
        elif len(ds_ids) == target_len:
            ds_ids = ds_ids
        elif len(ds_ids) < target_len:
            _, all_ds_ids = self.index.search(question_hidden_states, self.config.n_docs + 5)
            ds_ids = [doc_id for sublist in all_ds_ids for doc_id in sublist]
            ds_ids = [doc_id for doc_id in ds_ids if doc_id not in positive_ids]
            if len(ds_ids) >= target_len:
                ds_ids = random.sample(ds_ids, target_len)
            else:
                assert len(ds_ids) <= target_len

        batch_ds_ids, positive_ctx_indices = self._insert_positive_ids(
            ds_ids, positive_ids
        )

        # Convert batch_ds_ids to numpy array and retrieve embeddings from index
        batch_ds_ids = np.array(batch_ds_ids)
        ctx_emb = self.embeddings[batch_ds_ids]
        return ctx_emb, positive_ctx_indices, batch_ds_ids

    def retrieve_t1(self, question_hidden_states, positive_ids):
        # Returns a per-question list of candidates like [[pos, neg, neg, ...], [...], ...]
        _, all_ds_ids = self.index.search(question_hidden_states, self.config.n_docs + 10)
        all_ds_ids = all_ds_ids.tolist()

        if self.config.model_type == "ance":
            positive_ids = [int(p_id) for p_id in positive_ids]
        else:
            positive_ids = [int(p_id) - 1 for p_id in positive_ids]

        batch_ds_ids = []
        positive_ctx_indices = []
        offset = 0

        for i, row in enumerate(all_ds_ids):
            p_id = positive_ids[i] if i < len(positive_ids) else None
            row = [doc_id for doc_id in row if doc_id not in positive_ids]
            if p_id is not None:
                row.insert(0, p_id)
                positive_ctx_indices.append(offset)
            row_ds_ids = row[:self.config.n_docs]
            batch_ds_ids.extend(row_ds_ids)
            offset += len(row_ds_ids)

        batch_ds_ids = np.array(batch_ds_ids)
        ctx_emb = self.embeddings[batch_ds_ids]
        return ctx_emb, positive_ctx_indices, batch_ds_ids

    def eval_retrieve(self, question_hidden_states, positive_ids, retrive_n = 1000):
        # Similar to `retrieve`. For evaluating top-k (e.g., top 1000) retrieval quality,
        # we use the top `retrive_n` IDs and return the full retrive_I matrix instead of
        # batch_ds_ids.
        _, retrive_I = self.index.search(question_hidden_states, retrive_n)
        all_ds_ids = retrive_I[:, :self.config.n_docs + 10]
        ds_ids = list(all_ds_ids.flatten().tolist())

        # Convert positive_ids to integers and decrease by 1
        if self.config.model_type == "ance":
            positive_ids = [int(p_id) for p_id in positive_ids]
        else:
            positive_ids = [int(p_id) - 1 for p_id in positive_ids]
        ds_ids = [doc_id for doc_id in ds_ids if doc_id not in positive_ids]

        # Calculate target length for ds_ids after inserting positive_ids
        target_len = (
            len(positive_ids) * (self.config.n_docs - 1)
            + len(positive_ids)
            - len(set(positive_ids))
        )
        if len(ds_ids) > target_len:
            ds_ids = random.sample(ds_ids, target_len)
        elif len(ds_ids) == target_len:
            ds_ids = ds_ids
        elif len(ds_ids) < target_len:
            _, all_ds_ids = self.index.search(question_hidden_states, self.config.n_docs + 5)
            ds_ids = [doc_id for sublist in all_ds_ids for doc_id in sublist]
            ds_ids = [doc_id for doc_id in ds_ids if doc_id not in positive_ids]
            if len(ds_ids) >= target_len:
                ds_ids = random.sample(ds_ids, target_len)
            else:
                assert len(ds_ids) <= target_len

        batch_ds_ids, positive_ctx_indices = self._insert_positive_ids(
            ds_ids, positive_ids
        )

        # Convert batch_ds_ids to numpy array and retrieve embeddings from index
        batch_ds_ids = np.array(batch_ds_ids)
        ctx_emb = self.embeddings[batch_ds_ids]
        return ctx_emb, positive_ctx_indices, retrive_I

    def _insert_positive_ids(self, ds_ids, positive_ids):
        # remove repeated positive_ids
        unique_positive_ids = list(dict.fromkeys(positive_ids))
        insert_indices = sorted(random.sample(range(len(ds_ids) + 1), len(unique_positive_ids)))
        for i, p_id in zip(insert_indices, unique_positive_ids):
            ds_ids.insert(i, p_id)
        position_map = {p_id: idx for p_id, idx in zip(unique_positive_ids, insert_indices)}
        positive_ctx_indices = [position_map[p_id] for p_id in positive_ids]
        return ds_ids, positive_ctx_indices
    
    def get_top_ids(self, question_hidden_states, n_docs=5):
        _, ds_ids = self.index.search(question_hidden_states, n_docs)
        return ds_ids

@ray.remote
class RayRetriever(Retriever):
    def __init__(self, config):
        super().__init__(config)
    
    def get_dataset(self):
        return self.dataset
    
    def update_dataset(self, dataset):
        self.dataset = dataset


def print_memory(prefix=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**3  # GB
    gpu_mem = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{prefix}] CPU RSS: {cpu_mem:.2f} GB | GPU alloc: {gpu_mem:.2f} GB | reserved: {gpu_reserved:.2f} GB")
    else:
        print(f"[{prefix}] CPU RSS: {cpu_mem:.2f} GB (no GPU)")
    return cpu_mem, gpu_mem
