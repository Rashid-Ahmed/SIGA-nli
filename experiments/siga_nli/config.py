from typing import Optional
import torch
from pydantic import BaseModel
from pathlib import Path


class TrainingConfig(BaseModel):
    epochs: int = 3
    lr: float = 5e-6
    batch_size_per_device: int = 8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    weight_decay: float = 0
    evaluation_metric: str = "f1_score"

    @property
    def selected_device(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"


class ModelConfig(BaseModel):
    model_name_or_path: str = "microsoft/deberta-base-mnli"
    tokenizer_name: Optional[str] = "microsoft/deberta-base-mnli"
    model_checkpoint: str = Path("temp") / "model.ckpt"
    fast_tokenizer: bool = True


class DataConfig(BaseModel):
    train_data_dir: Path = Path.cwd().parent / Path("data") / "train_dataset.csv"
    id_test_data_dir: str = Path.cwd().parent / Path("data") / "test_id_dataset.csv"
    ood_test_data_dir: str = Path.cwd().parent / Path("data") / "test_ood_dataset.csv"
    max_token_length: Optional[int] = 256
    padding: bool = True
    truncation: bool = True
    num_labels: int = 3


class Config(BaseModel):
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
