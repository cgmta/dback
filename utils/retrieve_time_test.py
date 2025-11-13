import faiss
from pyserini.encode import AnceEncoder
from datasets import load_from_disk
from transformers import RobertaTokenizer
import torch
import time
import os
import numpy as np

def benchmark_faiss_retrieval(
    index_path: str,
    dataset_path: str,
    model_name: str = 'castorini/ance-msmarco-passage',
    split: str = 'train',
    batch_size: int = 32,
    max_length: int = 256,
    n_doc: int = 500,
    n_trials: int = 10,
):
    """
    Benchmark FAISS retrieval time for a given set of queries.

    Args:
        index_path (str): Path to the FAISS index file.
        dataset_path (str): Path to the Hugging Face dataset on disk.
        model_name (str): Hugging Face model name for query encoder.
        split (str): Dataset split to use ('train', 'dev', etc.).
        num_queries (int): Number of queries to test.
        batch_size (int): Batch size for encoding queries.
        max_length (int): Max token length for tokenizer.
    """
    assert os.path.exists(index_path), f"Index not found at: {index_path}"
    assert os.path.exists(dataset_path), f"Dataset not found at: {dataset_path}"

    # Load index and dataset
    print(f"Loading index from: {index_path}")
    index = faiss.read_index(index_path)

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    queries = dataset[split]['query'][:batch_size]

    # Load encoder and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    encoder = AnceEncoder.from_pretrained(model_name)

    inputs = tokenizer(
        queries,
        max_length=max_length,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        embeddings = encoder(inputs['input_ids']).cpu().numpy()

    print("Starting FAISS search...")
    start_time = time.time()
    _, I = index.search(embeddings, k=n_doc + 10)
    duration = time.time() - start_time
    
    print(f"Retrieved top-10 results for {batch_size} queries in {duration * 1000:.2f} ms.")
    print(f"Average time per query: {duration * 1000 / batch_size:.2f} ms.")
    
    total_time = 0.0
    for i in range(n_trials):
        queries = np.random.randn(batch_size, 768).astype(np.float32)
        start_time = time.time()
        _, I = index.search(queries, k=n_doc + 10)
        duration = time.time() - start_time
        total_time += duration
        print(f"Trial {i+1}: search time = {duration*1000:.2f} ms")
    avg_time_ms = (total_time / n_trials) * 1000
    
    print(f"Average search time over {n_trials} trials: {avg_time_ms:.2f} ms")

    return I

def main():
    index_path = './compress_index/index'  # Update with your index path
    dataset_path = './MS-MARCO/marco_hf/msmarco_train_dev'  # Update with your dataset path
    model_name = 'castorini/ance-msmarco-passage'  # Update with your model name

    benchmark_faiss_retrieval(
        index_path=index_path,
        dataset_path=dataset_path,
        model_name=model_name,
        batch_size=32,
        max_length=256
    )

if __name__ == "__main__":
    main()