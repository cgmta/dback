import os
import time
import math
import argparse
import hydra
import json
import logging
import numpy as np
from contextlib import nullcontext
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast, GradScaler


import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group

from model.model import DbackModel
from model.retriever import Retriever
from utils.data_utils import get_dataloader

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# set training precision
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = 'cuda'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else autocast(device_type=device_type, dtype=ptdtype)


def get_lr(it, args):
    if it < args.warmup_steps:
        return args.lr * it / args.warmup_steps
    if it > args.lr_decay_iters:
        return args.min_lr
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - args.warmup_steps) / (args.lr_decay_iters - args.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.lr - args.min_lr)

def get_dataloader(args):
    # manully load train and dev data
    nq_dataset = load_from_disk(args.train_data)

    # Create Distributed Samplers for train and validation sets
    train_sampler = None
    val_sampler = None
    shuffle_train = True    

    train_loader = DataLoader(
        nq_dataset["train"],
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4,
        sampler=train_sampler,
        shuffle=shuffle_train,
    )
    
    eval_split = "test" if "test" in nq_dataset else "dev"

    val_loader = DataLoader(
        nq_dataset[eval_split],
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4,
        sampler=val_sampler,
    )

    return train_loader, val_loader

def compute_mrr_and_recall(all_retrive_I, all_positive_ids, topk=(10, 1000)):
    import numpy as np
    I = np.concatenate(all_retrive_I, axis=0)
    positive_ids = sum([p.tolist() if isinstance(p, torch.Tensor) else p for p in all_positive_ids], [])  # flatten list of lists

    mrr_10 = 0.0
    acc_top_1000 = 0.0
    acc_top_10 = 0.0
    total = len(positive_ids)

    for i in range(total):
        retrieved = I[i]
        pos_id = int(positive_ids[i])

        top_10 = retrieved[:10]
        if pos_id in top_10:
            rank = top_10.tolist().index(pos_id) + 1
            mrr_10 += 1.0 / rank
            acc_top_10 += 1

        if pos_id in retrieved[:1000]:
            acc_top_1000 += 1

    mrr_10 /= total
    acc_top_1000 /= total
    acc_top_10 /= total
    return mrr_10, acc_top_1000, acc_top_10

def init_ddp(args):
    OmegaConf.set_struct(args, False)
    args.ddp = int(os.environ.get('RANK', -1)) != -1
    if args.ddp:
        init_process_group(backend=args.backend, init_method="env://")
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
        args.rank = int(os.environ.get('RANK', 0))
        args.offset = args.rank
        args.master = args.rank == 0
        args.device = f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu"
    else:
        args.master = True
        args.world_size = 1
        args.offset = 0
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.set_struct(args, True)
    return args

def save_args_to_json(args, filepath):
    def serialize_obj(obj):
        if isinstance(obj, torch.device):
            return str(obj)  # 将 torch.device 转为字符串
        return obj

    args_dict = {key: serialize_obj(value) for key, value in vars(args).items()}
    with open(filepath, "w") as f:
        json.dump(args_dict, f, indent=4)
        
def init_set(cfg: DictConfig):
    # get the config and args from hydra
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    args = argparse.Namespace(**config_dict)
    
    torch.manual_seed(1337)
    os.makedirs(args.out_dir, exist_ok=True)
    return args

def run_batch_step(model, retriever, batch, device, temperature):
    # prepare the batch
    positive_ids = batch["positive_id"]

    # according to the question, retrieve the context embeddings
    with torch.no_grad():
        q_emb = model.encode_query(batch["query"])
    rq_emb = q_emb.cpu().detach().numpy()
    ctx_emb, pctx_index, batch_ds_ids = retriever.retrieve(rq_emb, positive_ids)
    ctx_emb = torch.tensor(ctx_emb).to(device)
    pctx_index = torch.tensor(pctx_index).to(device)

    # calculate the loss
    loss, correct_p_count = model(
        q_emb,
        ctx_emb,
        pctx_index,
        temperature,
    )
    
    return loss, correct_p_count, batch_ds_ids

def evaluate_model(model, retriever, val_loader, device, temperature, bestv_rate, args):
    """
    Evaluate model on validation set and optionally save best checkpoint.

    Args:
        model: Your main dual-encoder / retriever model.
        retriever: Retriever object for context retrieval.
        val_loader: Validation DataLoader.
        device: torch.device("cuda" or "cpu").
        temperature (float): Temperature for similarity scaling.
        bestv_rate (float): Best validation correct rate so far.
        args: Config object containing options like eval_interval, wandb_log, out_dir, etc.

    Returns:
        bestv_rate (float): Updated best validation correct rate.
    """

    model.eval()
    val_loss, val_correct_p_count, val_total_p_count = 0.0, 0, 0 
    all_retrive_I, all_positive_ids =[], []
    with torch.no_grad():
        for batch in val_loader:
            positive_ids = batch["positive_id"]
            q_emb = model.encode_query(batch["query"])
            rq_emb = q_emb.cpu().detach().numpy()
            ctx_emb, pctx_index, retrive_I = retriever.eval_retrieve(
                rq_emb, positive_ids
            )
            ctx_emb = torch.tensor(ctx_emb).to(device)
            pctx_index = torch.tensor(pctx_index).to(device)
            # forward pass
            loss, correct_p_count = model(
                q_emb,
                ctx_emb,
                pctx_index,
                temperature,
            )

            # collect the loss and correct_p_count
            val_loss += loss.item()
            val_correct_p_count += correct_p_count.item()
            val_total_p_count += len(pctx_index)

            all_retrive_I.append(retrive_I)
            all_positive_ids.append(positive_ids)
            
    # log the validation loss and correct_p_rate
    model.train()
    mrr_10, acc_top_1000, acc_top_10 = compute_mrr_and_recall(all_retrive_I, all_positive_ids)
    print(f"acc_mrr_10: {mrr_10:.4f} val_hits_10: {acc_top_10:.4f} val_hits_1000: {acc_top_1000:.4f}")
    val_loss /= len(val_loader)
    val_c_rate = val_correct_p_count / val_total_p_count
    print(f"val_loss: {val_loss:.4f} val_correct_p_rate: {val_c_rate}")
 
    # ===== Save model if performance improved =====
    if mrr_10 > args.best_acc_mrr_10:
        args.best_acc_mrr_10 = mrr_10
        save_dir = os.path.join(args.out_dir, f"checkpoint_{mrr_10:.3f}")
        os.makedirs(save_dir, exist_ok=True)

        # Save model components
        model.q_encoder.save_pretrained(os.path.join(save_dir, "encoder"))

        # Save retriever index and embeddings
        retriever.save_index(save_dir)
        retriever.save_embeddings(save_dir)

        # Save training configuration
        args_path = os.path.join(save_dir, "config.yaml")
        OmegaConf.save(config=args, f=args_path)
        logging.info("New best model found — checkpoint saved.")
    return val_c_rate

@hydra.main(config_path="conf", config_name="train_ms", version_base=None)
def main(args: DictConfig):   
    args = init_ddp(args)
    torch.manual_seed(1337+args.offset)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    scaler = GradScaler()
    temperature = args.temperature
    global_iters = 0
    bestv_rate = 0.0

    # get the dataloader
    train_loader, val_loader = get_dataloader(args)

    # init model & retriver
    model = DbackModel(args)
    model = model.to(device)
    retriever = Retriever(args)
    
    # init optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, args.ctx_lr, (args.beta1, args.beta2), device_type)

    # logging
    if args.wandb_log:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    t0 = time.time()
    epoch_iters = len(train_loader)
    
    # training loop
    for epoch in range(args.epochs):
        # train_loader.sampler.set_epoch(epoch)
        for iter, batch in enumerate(train_loader):
            global_iters += 1
            # determine and set the learning rate for this iteration
            lr = get_lr(global_iters, args) if args.decay_lr else args.lr  
            for param_group in optimizer.param_groups[-2:]:
                param_group["lr"] = lr
                
            # ---- No gradient accumulation: step every batch ----
            # Ensure gradients are synced in DDP mode
            if args.ddp:
                model.require_backward_grad_sync = True

            loss, cp_count, batch_ds_ids = run_batch_step(model, retriever, batch, device, temperature)

            # backward with AMP (no accumulation)
            scaler.scale(loss).backward()

            # unscale gradients for clipping if needed
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # update context embeddings
            if args.update_ctx_emb:
                new_ctx_emb = np.array(model.ctx_emb.tolist())
                retriever.update_emb(batch_ds_ids, new_ctx_emb)

            lossf = loss.item()
            # use the tqdm to log the loss and correct_p_count
            if global_iters % args.log_interval == 0:
                # Logging the loss and time
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                log.info(
                    f"Epoch {epoch} Iter {iter}/{epoch_iters} Train loss: {lossf:.4f} Correct count: {cp_count} lr: {lr:.6f} Time: {dt*1000:.4f}s"
                )
                if args.wandb_log:
                    wandb.log(
                        {
                            "global_iters": global_iters,
                            "epoch": epoch,
                            "train_loss": lossf,
                            "train_c_count": cp_count,
                            "lr": lr,
                        }
                    )
                    
            # update the faiss index according to the new context embeddings
            if args.update_ctx_emb and global_iters % args.index_interval == 0:
                retriever.update_qhnsw_faiss()
            
            if args.eval and global_iters % args.eval_interval == 0:
                bestv_rate = evaluate_model(model, retriever, val_loader, device, temperature, bestv_rate, args)

    if args.ddp:
        destroy_process_group()
        
if __name__ == "__main__":
    main()