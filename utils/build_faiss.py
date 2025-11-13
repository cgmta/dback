import numpy as np
import faiss
import os
import logging

logger = logging.getLogger()

def build_flat_index_from_npy(
    npy_path: str,
    index_save_path: str = None,
    metric: str = "ip",
    normalize: bool = False
) -> faiss.Index:
    """
    Load embeddings from a .npy file, build a FAISS index (L2 or Inner Product),
    and optionally save it to disk.

    Args:
        npy_path (str): Path to the .npy file containing embeddings.
        index_save_path (str, optional): Path to save the FAISS index.
        metric (str): "ip" for inner product or "l2" for L2 distance. Default: "ip".
        normalize (bool): Whether to apply L2 normalization (for cosine similarity behavior).
    
    Returns:
        faiss.Index: The built FAISS index.
    """
    # --- Load embeddings ---
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Embedding file not found: {npy_path}")

    embeddings = np.load(npy_path)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # --- Optional normalization (cosine similarity) ---
    if normalize:
        faiss.normalize_L2(embeddings)

    # --- Create index ---
    dim = embeddings.shape[1]
    metric = metric.lower()
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'.")

    # --- Add all embeddings at once ---
    index.add(embeddings)
    print(f"FAISS index built successfully! Total entries: {index.ntotal}, Dimension: {dim}")

    # --- Optionally save ---
    if index_save_path:
        os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
        faiss.write_index(index, index_save_path)
        print(f"FAISS index saved to {index_save_path}")

    return index

def build_ivfpq_index_from_npy(
    file_path: str,
    save_path: str
):
    """
    Build an FAISS IndexIVFPQ index from a large .npy file and save it.

    Args:
        file_path (str): Path to the .npy file containing the vectors.
        save_path (str): Path to save the FAISS index. If None, the index is not saved.

    Returns:
        faiss.IndexIVFPQ: The constructed FAISS index.
    """
    # Load the vectors in memory-mapped mode
    embeddings = np.load(file_path)
    d = embeddings.shape[1]  # Dimension of vectors
    quantizer = faiss.IndexHNSWFlat(d, 128, faiss.METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = 200
    quantizer.hnsw.efSearch = 128

    ivf_index = faiss.IndexIVFPQ(quantizer, d, 4096, 128, 8, faiss.METRIC_INNER_PRODUCT)
    ivf_index.nprobe = 128
    ivf_index.own_fields = True
    quantizer.this.disown()

    train_size = 262144

    print("Training FAISS quantizer...")
    ivf_index.train(embeddings[:train_size]) 
    print("FAISS quantizer trained.")

    print(f"Adding {len(embeddings)} embeddings to the index...")
    ivf_index.add(embeddings)
    # Save the index
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(ivf_index, save_path)
        print(f"Index saved to {save_path}")

    return ivf_index


# IndexFlatIP index method
# import faiss  
# dataset.add_faiss_index(column="embeddings", metric_type=faiss.METRIC_INNER_PRODUCT)

if __name__ == "__main__":
    # Path to the large .npy file containing the vectors
    file_path = "./train_output/ance_ms/checkpoint_mrr_10_0.263/embeddings.npy"
    # Path to save the FAISS index
    save_path = "./train_output/ance_ms/checkpoint_mrr_10_0.263/flat_index.faiss"
    
    # Construct the FAISS index
    build_flat_index_from_npy(file_path, save_path)