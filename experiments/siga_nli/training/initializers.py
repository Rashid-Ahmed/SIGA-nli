from siga_nli.config import Config
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch


def initialize_tokenizer(config: Config):
    auto_config = AutoConfig.from_pretrained(config.model.model_name_or_path, num_labels=config.data.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name)

    return auto_config, tokenizer


def initialize_model(
        config: Config,
        auto_config: AutoConfig.from_pretrained,

):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        config=auto_config,
    ).to(config.training.selected_device)
    return model


def load_model(
        config: Config,
        auto_config: AutoConfig.from_pretrained,
):
    model = initialize_model(config, auto_config)
    model.load_state_dict(
        torch.load(config.model.model_checkpoint))

    return model
