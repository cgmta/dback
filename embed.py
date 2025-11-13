import torch
import os
import torch.multiprocessing as mp
from dataclasses import dataclass
from utils.context_embed_utils import embed_update, add_compress_index, concat_dataset, embed_update_ance

@dataclass
class ValidationConfig:
    # save path
    base_dir: str = "./dataset/wikidir_dpr_single_base_line"
    # origin dataset
    wiki_noemb: str =  "./dataset/wiki_base_line/wiki_noemb"
    # context encoder and tokenizer
    ctx_encoder: str = "facebook/dpr-context_encoder-single-nq-base"
    ctx_tokenizer: str = "facebook/dpr-context_encoder-single-nq-base"
    
def main():
    mp.set_start_method("spawn", force=True)
    hparams = ValidationConfig()

    processes = []
    world_size = torch.cuda.device_count() 
    available_gpus = [torch.device(f"cuda:{i}") for i in range(world_size)]

    # Initialize and start embedding update processes
    for rank in range(world_size):
        print(f"Initializing embedding calculation process rank {rank}")
        device = available_gpus[rank]
        p = mp.Process(
            target=embed_update,
            args=(
                hparams,
                world_size,
                rank,
                device,
            ),
        )
        processes.append(p)

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()  # Ensure each process finishes

    # Run index compression after all embedding updates are complete
    shard_dir = os.path.join(hparams.base_dir, "wiki/shard_dir")
    dataset = concat_dataset(shard_dir)
    
    index_path = os.path.join(hparams.base_dir, "wiki/index/compress_index.faiss")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    dataset = add_compress_index(dataset, index_path)
    
    dataset.drop_index('embeddings')
    wiki_embed = os.path.join(hparams.base_dir, "wiki/wiki_noindex")
    os.makedirs(wiki_embed, exist_ok=True)
    columns_to_remove = [col for col in dataset.column_names if col != 'embeddings']
    dataset = dataset.remove_columns(columns_to_remove)
    dataset.save_to_disk(wiki_embed)

if __name__ == "__main__":
    main()