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
from torch.distributed import init_process_group, destroy_process_group

from model.model import DbackModel
from model.retriever import Retriever
from utils.validation_utils import calculate_matches
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
    """
    Runs one step of forward pass with retrieval and loss computation.

    Args:
        model: The dual-encoder or retriever model.
        retriever: The actor or module that performs context retrieval.
        batch (dict): A batch of input data with keys:
                      "input_ids", "token_type_ids", "attention_mask", "positive_ids"
        device (torch.device): The target device (e.g., torch.device("cuda")).
        temperature (float): Temperature value used in softmax scaling.

    Returns:
        loss (torch.Tensor): The computed loss for the batch.
        correct_p_count (int): Number of correctly retrieved positive contexts.
        batch_ds_ids (List): Dataset IDs for each instance in the batch.
    """
    # Prepare the batch
    question_input_ids = batch["input_ids"].to(device)
    question_type_ids = batch["token_type_ids"].to(device)
    question_attention_mask = batch["attention_mask"].to(device)
    positive_ids = batch["positive_ids"]

    # Encode the question
    with ctx:
        q_emb = model.get_q_emb(question_input_ids, question_type_ids, question_attention_mask)
    re_emb = q_emb.cpu().detach().numpy()

    # Retrieve top contexts using retriever
    ctx_emb, pctx_index, batch_ds_ids = retriever.retrieve(re_emb, positive_ids)

    # Convert retrieved context embeddings and indices to torch tensors
    ctx_emb = torch.tensor(ctx_emb).to(device)
    pctx_index = torch.tensor(pctx_index).to(device)

    # Forward pass with model
    with ctx:
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
    val_loss = 0.0
    val_correct_p_count = 0
    val_total_p_count = 0

    with torch.no_grad():
        for val_batch in val_loader:
            # forward
            loss, correct_p_count, _ = run_batch_step(
                model,
                retriever,
                val_batch, 
                device,
                temperature,
            )

            # collect metrics
            val_loss += loss.item()
            val_correct_p_count += correct_p_count
            val_total_p_count += len(val_batch["positive_ids"])

    # average over validation set
    val_loss /= len(val_loader)
    val_c_rate = val_correct_p_count / val_total_p_count

    # logging
    model.train()
    logging.info(f"val_loss: {val_loss:.4f}  val_correct_p_rate: {val_c_rate:.4f}")

    if getattr(args, "wandb_log", False):
        import wandb
        wandb.log({
            "val_loss": val_loss,
            "val_correct_p_rate": val_c_rate,
        })

    # Save best checkpoint
    if val_c_rate > bestv_rate:
        bestv_rate = val_c_rate
        save_path = os.path.join(args.out_dir, f"val_encoder_{bestv_rate:.4f}")
        model.q_encoder.save_pretrained(save_path+"/qencoder")
        torch.save(model.cor_net.state_dict(), os.path.join(save_path, "cor_model.bin"))
        logging.info(f"✅ Saved best model checkpoint to {save_path}")
    return bestv_rate

def test_model(model, retriever, test_loader, answers, device, max_match, args):
    """
    Evaluate the model on the validation set and save the checkpoint
    if the current performance surpasses the previous best.

    Args:
        model: The main model containing q_encoder and cor_net.
        retriever: Retriever object with get_dataset(), get_top_ids(), save_index(), etc.
        test_loader: DataLoader for the validation set.
        answers: Ground truth answers for validation questions.
        args: Configuration namespace containing paths and settings (e.g., out_dir).
        device: torch.device to run computations on.
        max_match: Best top-1 hit count so far (will be updated and returned).

    Returns:
        Updated max_match (new best value if improved).
    """
    logging.info("Start testing...")
    wiki_dataset = retriever.get_dataset()

    # ===== Extract all question embeddings =====
    model.eval()
    all_q_embs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            q_emb = model.get_q_emb(
                batch["input_ids"],
                batch["token_type_ids"],
                batch["attention_mask"],
            )
            all_q_embs.append(q_emb)

    # Concatenate all question embeddings
    all_embs = torch.cat(all_q_embs).cpu().numpy()

    # ===== Retrieve top-k document IDs for each question =====
    top_ids = retriever.get_top_ids(all_embs).tolist()

    # ===== Compute top-k hit statistics =====
    match_stats = calculate_matches(
        wiki_dataset,
        answers,
        top_ids,
        2,
        match_type="string",
    )
    top_k_hits = match_stats.top_k_hits
    top_k_hits_rate = [v / len(top_ids) for v in top_k_hits]

    logging.info("Validation results: top k documents hits %s", top_k_hits)
    logging.info("Validation results: top k documents hits accuracy %s", top_k_hits_rate)

    # ===== Save model if performance improved =====
    if top_k_hits[0] > max_match:
        max_match = top_k_hits[0]
        save_dir = os.path.join(args.out_dir, f"checkpoint_{max_match}")
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
 
    return max_match

@hydra.main(config_path="conf", config_name="train_nq", version_base=None)
def main(args: DictConfig):   
    args = init_ddp(args)
    torch.manual_seed(1337+args.offset)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    scaler = GradScaler()
    temperature = args.temperature
    global_iters = 0
    bestv_rate = 0.0
    max_match = args.max_match

    # get the dataloader
    train_loader, val_loader, test_loader, answers = get_dataloader(args)

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
            
            if args.eval and global_iters % args.test_interval == 0:
                bestv_rate = evaluate_model(model, retriever, val_loader, device, temperature, bestv_rate, args)

    max_match = test_model(model, retriever, test_loader, answers, device, max_match, args)

    if args.ddp:
        destroy_process_group()
        
if __name__ == "__main__":
    main()