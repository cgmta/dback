import os
import ast
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_from_disk, load_dataset

from transformers import AutoTokenizer, DataCollatorWithPadding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode(examples, tokenizer):
    inputs = tokenizer(
        examples["question"],
        truncation=True,
        max_length=256,
    )
    if "positive_ctxs" in examples:
        inputs["positive_ids"] = [p[0]["passage_id"] for p in examples["positive_ctxs"]]
    elif "positive_ids" in examples:
        inputs["positive_ids"] = [p[0] for p in examples["positive_ids"]]
    else:
        inputs["positive_ids"] = [0] * len(examples["question"])
    return inputs

def collate_fn(batch, data_collator):
    # use data_collator do padding
    input_batch = [
        {k: v for k, v in item.items() if k != "positive_ids"} for item in batch
    ]
    batch_dict = data_collator(input_batch)
    # positive_ids add to collated batch
    batch_dict["positive_ids"] = [item["positive_ids"] for item in batch]
    return batch_dict

def parse_answers(example):
    example["answers"] = ast.literal_eval(example["answers"])
    return example

def t_encode(example, tokenizer):
    return tokenizer(example["question"], truncation=True, max_length=256)

def get_dataloader(args):
    tokenizer = AutoTokenizer.from_pretrained(args.q_tokenizer) # manully load train and test data
    # manully load train and test data
    dataset = load_from_disk(args.train_data)
    dataset = dataset.map(
        partial(encode, tokenizer = tokenizer), batched=True, remove_columns=dataset["train"].column_names
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Create Distributed Samplers for train and validation sets
    train_sampler = None
    val_sampler = None
    shuffle_train = True    

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4,
        sampler=train_sampler,
        shuffle=shuffle_train,
        collate_fn=partial(collate_fn, data_collator = data_collator),
    )
    
    eval_split = "test" if "test" in dataset else "dev"

    val_loader = DataLoader(
        dataset[eval_split],
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=4,
        sampler=val_sampler,
        collate_fn=partial(collate_fn, data_collator = data_collator),
    )

    if args.test_data is None:
        return train_loader, val_loader, None, None
    test = load_dataset(
        "csv",
        data_files=args.test_data,
        delimiter="\t",
        column_names=["question", "answers"],
    )
    test = test.map(parse_answers)
    answers = test["train"]["answers"]
    test = test.map(
        partial(t_encode, tokenizer=tokenizer), batched=True, remove_columns=test["train"].column_names
    )
    test_loader = DataLoader(
        test["train"], batch_size=args.batch_size, num_workers=4, collate_fn=data_collator
    )
    return train_loader, val_loader, test_loader, answers