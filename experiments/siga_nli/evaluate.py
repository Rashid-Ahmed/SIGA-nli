import logging
from siga_nli.config import Config
from siga_nli.training.initializers import load_model, initialize_tokenizer
from siga_nli.training.pipeline import evaluate_model
from siga_nli.data.processing import load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(config: Config):
    auto_config, tokenizer = initialize_tokenizer(config)
    model = load_model(config, auto_config)
    id_testing_dataset, _ = load_data(config.data.id_test_data_dir, first_split=1)
    ood_testing_dataset, _ = load_data(config.data.ood_test_data_dir, first_split=1)
    id_evaluation_results = evaluate_model(id_testing_dataset, model, tokenizer, config.training.evaluation_metric,
                                           config)
    ood_evaluation_results = evaluate_model(ood_testing_dataset, model, tokenizer, config.training.evaluation_metric,
                                            config)
    logger.info(id_evaluation_results[config.training.evaluation_metric])
    logger.info(ood_evaluation_results[config.training.evaluation_metric])
