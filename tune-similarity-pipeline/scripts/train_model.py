import os
import sys
from src.utils.config import load_config
from src.training.trainer import Trainer

def main():
    # Load configuration
    config = load_config('configs/config.yaml')

    # Initialize the Trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()