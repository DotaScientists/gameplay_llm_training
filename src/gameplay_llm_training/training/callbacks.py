from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, Trainer
from loguru import logger



class LRCallback(TrainerCallback):

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        logger.info(f"{state.log_history}")