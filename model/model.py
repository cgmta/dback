"""
Define the model for the application
"""

import inspect
import torch.nn.init as init
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import DPRQuestionEncoder, RobertaTokenizer, DPRQuestionEncoderTokenizer
from pyserini.encode import AnceEncoder


class EmbModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        if config.model_type == "ance":
            self.q_encoder = AnceEncoder.from_pretrained(config.q_name_or_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(config.q_tokenizer or config.q_name_or_path,
                                                              clean_up_tokenization_spaces=True)
            self.tokenizer.do_lower_case = True
        elif config.model_type == "dpr":
            self.q_encoder = DPRQuestionEncoder.from_pretrained(config.q_name_or_path)
            self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(config.q_tokenizer or config.q_name_or_path,
                                                              clean_up_tokenization_spaces=True)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def get_q_emb(self, input_ids, attention_mask=None, token_type_ids=None):
        question_emb = self.q_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return question_emb.pooler_output if hasattr(question_emb, "pooler_output") else question_emb

    def encode_query(self, query: list[str]):
        if self.config.model_type == "ance":
            inputs = self.tokenizer(
                query,
                max_length=256,
                padding='longest',
                truncation=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
        else:
            inputs = self.tokenizer(
                query,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
        inputs.to(self.device)
        question_emb = self.q_encoder(**inputs)
        return question_emb
    
    def calc(self, q_emb, ctx_emb, pctx_index, temperature, loss_scale = None):
        """
        Computes nll loss for the given lists of question and ctx vectors.
        """
        scores = torch.matmul(q_emb, torch.transpose(ctx_emb, 0, 1))
        if len(q_emb.size()) > 1:
            q_num = q_emb.size(0)
            scores = scores.view(q_num, -1)
        scaled_scores = scores / temperature
        softmax_scores = F.log_softmax(scaled_scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            pctx_index.clone().detach().to(softmax_scores.device),
            reduction="mean",
        )

        _, max_idxs = torch.max(softmax_scores, 1)
        correct_p_count = (
            max_idxs == pctx_index.clone().detach().to(max_idxs.device)
        ).sum()

        if loss_scale:
            loss.mul_(loss_scale)

        return loss, correct_p_count

class DbackModel(EmbModel):
    def __init__(self, config):
        super().__init__(config)
        if config.update_ctx_emb:
            self.ctx_emb = nn.Parameter(torch.randn(config.batch_size * config.n_docs, config.n_embd))
        else:
            self.ctx_emb = None

    def forward(self, q_emb, ctx_emb, pctx_index, temperature=1, **kwargs):
        if self.config.update_ctx_emb:
            assert ctx_emb.shape == self.ctx_emb.shape
            with torch.no_grad():
                self.ctx_emb.copy_(ctx_emb)
            ctx_emb = self.ctx_emb  # Use the updated context embeddings
        return self.calc(q_emb, ctx_emb, pctx_index, temperature, **kwargs)


    def configure_optimizers(self, weight_decay, learning_rate, ctx_learning_rate, betas, device_type):
        optim_groups = [
            {'params': [self.ctx_emb], 'weight_decay': 0.0, 'lr': ctx_learning_rate}, 
            {'params': [p for n, p in self.q_encoder.named_parameters() if 'bias' not in n and 'LayerNorm.weight' not in n], 'weight_decay': weight_decay},  
            {'params': [p for n, p in self.q_encoder.named_parameters() if 'bias' in n or 'LayerNorm.weight' in n], 'weight_decay': 0.0}  
        ] 
        if not self.config.update_ctx_emb:
            optim_groups = optim_groups[1:]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
