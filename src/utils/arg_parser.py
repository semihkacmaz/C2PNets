import argparse
import yaml
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(description='Hybrid Piecewise Polytropic or Tabulated EoS-Based C2P Learner Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/nnc2ps.yaml',
        help='Path to default config file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['NNC2PS', 'NNC2PL', 'NNC2P_Tabulated'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs from config'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate from config'
    )
    return parser

def parse_args_and_config():
    parser = create_parser()
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with CLI arguments
    if args.model:
        config['model']['name'] = args.model
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate

    return args, config
