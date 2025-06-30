import os
import logging
import torch
from omegaconf import OmegaConf
import lightning.pytorch as pl
import nemo.collections.asr as nemo_asr
from nemo.utils import model_utils
from nemo.utils.exp_manager import exp_manager

# Setup logger
logger = logging.getLogger("ASRFineTuner")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class ASRFineTuner:
    def __init__(self, model_path, train_manifest, val_manifest, output_dir, learning_rate=1e-5, max_epochs=3, batch_size=4):
        self.model_path = model_path
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.cfg = self.create_config()

    def create_config(self):
        model_config = {
            'train_ds': {
                'manifest_filepath': self.train_manifest,
                'sample_rate': 16000,
                'batch_size': self.batch_size,
                'shuffle': True,
                'trim_silence': True,
                'max_duration': 20.0,
                'min_duration': 0.1,
                'normalize_transcripts': True,
            },
            'validation_ds': {
                'manifest_filepath': self.val_manifest,
                'sample_rate': 16000,
                'batch_size': self.batch_size,
                'shuffle': False,
                'normalize_transcripts': True,
            },
            'optim': {
                'name': 'adamw',
                'lr': self.learning_rate,
                'weight_decay': 1e-3,
                'sched': {
                    'name': 'CosineAnnealing',
                    'warmup_steps': 1000,
                    'min_lr': 1e-9,
                    'max_steps': None,
                }
            }
        }

        trainer_config = {
            'devices': 1,
            'accelerator': 'gpu',
            'max_epochs': self.max_epochs,
            'accumulate_grad_batches': 1,
            'gradient_clip_val': 1.0,
            'precision': 16,
            'check_val_every_n_epoch': 1,
            'enable_checkpointing': False,
            'logger': False,
            'log_every_n_steps': 10,
            'val_check_interval': 0.25,
        }

        exp_manager_config = {
            'exp_dir': self.output_dir,
            'name': 'parakeet_finetune_model',
            'create_tensorboard_logger': True,
            'create_checkpoint_callback': True,
            'checkpoint_callback_params': {
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3,
                'save_last': True,
                'filename': '{epoch:02d}-{val_wer:.4f}',
                'auto_insert_metric_name': False,
            },
            'resume_if_exists': False,
            'resume_ignore_no_checkpoint': True,
        }

        cfg = {
            'model': model_config,
            'trainer': trainer_config,
            'exp_manager': exp_manager_config,
            'init_from_nemo_model': self.model_path
        }

        return OmegaConf.create(cfg)

    def load_model(self, trainer):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        model = nemo_asr.models.ASRModel.restore_from(
            restore_path=self.cfg.init_from_nemo_model,
            map_location=device
        )
        model.set_trainer(trainer)
        logger.info("Model loaded and trainer set.")
        return model

    def setup_dataloaders(self, model):
        logger.info("Setting up data loaders...")
        cfg_dict = model_utils.convert_model_config_to_dict_config(self.cfg)
        model.setup_training_data(cfg_dict.model.train_ds)
        model.setup_multiple_validation_data(cfg_dict.model.validation_ds)
        logger.info("Data loaders are ready.")
        return model

    def train(self):
        logger.info("Starting training pipeline...")

        trainer = pl.Trainer(**self.cfg.trainer)
        logger.info("Trainer created.")

        exp_manager(trainer, self.cfg.exp_manager)
        logger.info("Experiment manager configured.")

        model = self.load_model(trainer)
        model = self.setup_dataloaders(model)
        model.setup_optimization(self.cfg.model.optim)

        logger.info(f"Training for {self.cfg.trainer.max_epochs} epochs...")
        trainer.fit(model)

        output_model_path = os.path.join(self.output_dir, 'parakeet_finetuned.nemo')
        model.save_to(output_model_path)
        logger.info(f"Fine-tuned model saved to: {output_model_path}")
        logger.info("Training complete.")

if __name__ == "__main__":
    tuner = ASRFineTuner(
        model_path="./model/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        train_manifest="../data/springlab-asr-task-wavs/train_manifest.json",
        val_manifest="../data/springlab-asr-task-wavs/validation_manifest.json",
        output_dir="./model/parakeet-spring-lab-asr-task-wavs-finetuned-model",
        learning_rate=1e-5,
        max_epochs=3,
        batch_size=4
    )
    tuner.train()