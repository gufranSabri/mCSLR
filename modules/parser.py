import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--work-dir', default="work_dir/test")
    parser.add_argument('--mode', default="train")
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--config', type=str, default='configs/combined.yaml')

    return parser