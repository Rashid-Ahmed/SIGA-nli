import logging
from pathlib import Path
import os
from siga_nli.config import Config
from siga_nli.data.processing import load_data
from siga_nli.training.initializers import initialize_model, initialize_tokenizer
from siga_nli.training.pipeline import train_model, evaluate_model
import torch
import pandas as pd

pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(output_path: Path, config: Config):
    train_dataset, validation_dataset = load_data(config.data.train_data_dir)
    auto_config, tokenizer = initialize_tokenizer(config)

    model = initialize_model(config, auto_config)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=config.training.weight_decay,
                                  betas=(config.training.adam_beta1, config.training.adam_beta2),
                                  eps=config.training.adam_epsilon, lr=config.training.lr)
    criterion = torch.nn.CrossEntropyLoss().to(config.training.selected_device)
    model = train_model(train_dataset, model, optimizer, criterion, tokenizer, config)
    evaluation_results = evaluate_model(validation_dataset, model, tokenizer, config.training.evaluation_metric, config)
    logger.info(evaluation_results[config.training.evaluation_metric])

    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
    tokenizer.save_vocabulary(os.path.join(output_path, "tokenizer"))
    torch.save(model.state_dict(),os.path.join(output_path, "model.ckpt"))
