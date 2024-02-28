from siga_nli.config import Config
from siga_nli.data.dataloader import NLIdataloader
import torch
from sklearn.metrics import f1_score
import numpy as np


def train_model(nli_df, model, optimizer, criterion, tokenizer, config: Config):
    for epoch in range(config.training.epochs):
        epoch_loss = 0
        dataloader = NLIdataloader(nli_df, tokenizer, config)
        for idx in range(len(dataloader)):
            embeddings, targets = dataloader[idx]
            embeddings = embeddings.to(config.training.selected_device)
            targets = targets.to(config.training.selected_device)

            optimizer.zero_grad()
            outputs = model(**embeddings).logits
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += outputs.shape[0] * loss.item()

    return model


def evaluate_model(nli_df, model, tokenizer, evaluation_metric: str, config: Config):
    predicted_labels = torch.empty(0, device=torch.device("cpu"))
    true_labels = []
    dataloader = NLIdataloader(nli_df, tokenizer, config)
    with torch.no_grad():
        for idx in range(len(dataloader)):
            embeddings, targets = dataloader[idx]
            embeddings = embeddings.to(config.training.selected_device)
            targets = targets.tolist()

            outputs = model(**embeddings).logits
            predicted_labels = torch.cat((predicted_labels, outputs.detach().cpu()), 0)
            true_labels.extend(targets)

    predicted_labels = predicted_labels.argmax(dim=1)
    predicted_labels = predicted_labels.numpy()
    true_labels = np.array(true_labels)

    if evaluation_metric == "accuracy":
        evaluation_value = (true_labels == predicted_labels).sum() / len(true_labels)
    else:
        classwise_f1_scores = {}
        for cls in [0, 1, 2]:
            true_cls = [1 if lbl == cls else 0 for lbl in true_labels]
            predicted_cls = [1 if lbl == cls else 0 for lbl in predicted_labels]

            f1 = f1_score(true_cls, predicted_cls)
            classwise_f1_scores[cls] = f1
        evaluation_value = classwise_f1_scores
    model_results = {"predicted": predicted_labels, "true": true_labels, evaluation_metric: evaluation_value}
    return model_results
