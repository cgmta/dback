# Improving the Accuracy of Dense Retrieval on Quantized Indexes via Gradient Optimization of the Target Embeddings

## 🚀 Overview

This paper, accepted as an Oral Presentation at AAAI 2026, introduces a novel training framework designed to improve dense retrieval accuracy under quantized index structures.

Our method enhances both efficiency and accuracy for quantized retrieval systems through three main components:

- Direct Gradient Updates on Cached Target Embeddings
- Similarity-Guided Large-Scale Negative Sampling
- Quantization-Aware Optimization for Scalable Dense Retrieval

## 📦 Datasets

### Training and Evaluation Data

The training datasets can be directly downloaded from Hugging Face:

- **Natural Questions (NQ)** → [`coign/dback_nq`](https://huggingface.co/datasets/coign/dback_nq)
- **TriviaQA** → [`coign/dback_trivia`](https://huggingface.co/datasets/coign/dback_trivia)
- **MS MARCO** → [`coign/dback_msmarco`](https://huggingface.co/datasets/coign/dback_msmarco)

### Passages

#### Wikipedia Passages

```bash
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

#### MS MARCO Passages

```bash
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
```

> ⚠️ All datasets are formatted using the [Hugging Face Datasets](https://github.com/huggingface/datasets) library.
> Please ensure the data are converted to that format before training.

### Passage Embeddings

You can obtain **passage embeddings** in two ways:

1. Build embeddings manually

    To manually generate passage embeddings:

    ```bash
    python embed.py
    ```

2. Extract from Released Flat Indexes

    Below are several publicly available indexes and their corresponding pretrained models:

    • **DPR Single (Wikipedia)**

    Download the index:

    ```bash
    wget https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index.dpr
    ```

    The corresponding model used to build this index:

    ```python
    from transformers import DPRContextEncoder

    encoder = DPRContextEncoder.from_pretrained("facebook/dpr-context_encoder-single-nq-base")
    ```

    • **ANCE Multi (Wikipedia)**

    Download the index:

    ```bash
    wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-wikipedia-ance_multi-bf-20210224-060cef.tar.gz
    ```

    The corresponding model used to build this index:

    ```python
    from pyserini.encode import AnceEncoder

    encoder = AnceEncoder('castorini/ance-dpr-question-multi')
    ```

    • **ANCE (MS MARCO)**

    Download the index:

    ```bash
    wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/dindex-msmarco-passage-ance-bf-20210224-060cef.tar.gz
    ```

    The corresponding model used to build this index:

    ```python
    from pyserini.encode import AnceEncoder

    encoder = AnceEncoder('castorini/ance-msmarco-passage')
    ```

    Extract Embeddings via FAISS

    Once the index is downloaded, you can extract passage embeddings as NumPy arrays using FAISS:

    ```python
    import faiss
    import numpy as np

    # Load the FAISS index
    index = faiss.read_index("your_index.index")

    # Convert FAISS internal buffer to NumPy array
    embeddings = faiss.vector_float_to_array(index.xb).reshape(index.ntotal, index.d)

    print(embeddings.shape)
    # Output: (num_passages, embedding_dim)
    ```

### Passage Indexes

To build compressed FAISS indexes (e.g., IVFPQ, HNSW-PQ):

```bash
python utils/build_faiss.py
```

## 🏋️ Training

Once the data, embeddings, and index files are ready, start training:

```bash
python train.py
python train_ms.py
```

Training configurations (e.g., datasets, paths, hyperparameters) are stored in the `conf/` directory.

## 🧪 Citation

If you use or reference this work, please cite our paper:

```bibtex
@inproceedings{tan2026quantized,
  title     = {Improving the Accuracy of Dense Retrieval on the Quantized Indexes via Gradient Optimization of Target Embeddings},
  author    = {Tan, Cong and Shao, Yongqi and Huo, Hong and Fang, Tao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  note      = {Oral Presentation},
  institution = {Shanghai Jiao Tong University}
}
```

⭐ If you find this repository helpful, please consider giving it a star!
