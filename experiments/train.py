import argparse
import os
import sys

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train Spectral Emergent AI model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to config file")
    parser.add_argument('--data', type=str, default='synthetic', help="Dataset type: 'synthetic' or 'physionet'")
    args = parser.parse_args()
    
    trainer = Trainer(args.config, args.data)
    trainer.fit()

if __name__ == '__main__':
    main()
