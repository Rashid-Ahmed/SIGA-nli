from torch.utils.data import Dataset
from siga_nli.config import Config
import torch
import numpy as np


class NLIdataloader(Dataset):
    def __init__(self, nli_df, tokenizer, config: Config):
        self.nli_df = nli_df
        self.batch_size = config.training.batch_size_per_device
        self.len_dataset = len(nli_df) // self.batch_size
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, index):
        offset = index * self.batch_size
        batch = self.nli_df[offset: offset + self.batch_size]

        context = batch["premise"].tolist()
        hypothesis = batch["statement"].tolist()
        targets = batch["label"]

        targets[targets == "contradiction"] = 0
        targets[targets == "neutral"] = 1
        targets[targets == "entailment"] = 2

        embeddings = self.tokenizer(
            context,
            hypothesis,
            padding=self.config.data.padding,
            truncation=self.config.data.truncation,
            max_length=self.config.data.max_token_length,
            return_tensors="pt",
        )
        targets = torch.from_numpy(np.array(targets.astype(int))).type(torch.LongTensor)
        return embeddings, targets
