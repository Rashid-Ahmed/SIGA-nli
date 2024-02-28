# Instructions for finetuning and evaluating models

1) python 3.9.6 installed on the system.
2) Poetry installed on the system.
3) Go to experiments and run "poetry lock" and then "poetry install", this will create a .venv with all the required dependencies installed
4) By default pytorch cpu is installed, you will have to install pytorch gpu manually inside the virtual environment.
5) To finetune a model, run python cli.py train <output_model_path>
6) To evaluate a model, run python cli.py evaluate (evaluate by default expects a temp folder in the experiments directory that contains model checkpoint, for further information go to config.py -> ModelConfig -> model_checkpoint)
7) All the hyperparameters can be found in experiments -> siga_nli -> config.py
