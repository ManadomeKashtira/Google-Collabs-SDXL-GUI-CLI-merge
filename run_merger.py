import os
import sys
import argparse
from Test_merge import ModelMerger
from seed_utils import SeedManager

def create_parser():
    parser = argparse.ArgumentParser(description="Advanced AI Model Merger")
    
    # Previous arguments remain the same
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        help='2 or 3 model paths to merge'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for merged model'
    )
    
    # Add new seed-related arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: auto-generated)'
    )
    
    parser.add_argument(
        '--save-seed',
        action='store_true',
        help='Save seed information in metadata'
    )
    
    parser.add_argument(
        '--seed-info',
        action='store_true',
        help='Display detailed seed information'
    )
    
    # Add other args back
    parser.add_argument(
        '--method',
        default='weighted_sum',
        choices=ModelMerger.MERGE_METHODS.keys(),
        help='Merge method'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha ratio (for first model)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.0,
        help='Beta ratio (for third model, if used)'
    )
    
    parser.add_argument(
        '--precision',
        default='fp16',
        choices=ModelMerger.PRECISION_TYPES.keys(),
        help='Model precision'
    )
    parser.add_argument(
        '--vae-source',
        default='first',
        choices=['first', 'second', 'last'],
        help='VAE source model'
    )
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Initialize seed manager
    seed_manager = SeedManager()
    used_seed = seed_manager.set_seed(args.seed)
    
    if args.seed_info:
        print(f"\nUsing seed: {used_seed}")
        if args.seed is None:
            print("(Auto-generated seed)")
    
    config = {
        "model_paths": args.models,
        "output_path": args.output,
        "merge_method": args.method,
        "precision": args.precision,
        "merge_seed": used_seed,
        "alpha": args.alpha,
        "vae_source": args.vae_source,
        "beta": args.beta,
        "seed_manager": seed_manager
    }
    
    merger = ModelMerger(config)
    success = merger.merge()
    
    if success and args.save_seed:
        print("\nSeed information saved in model metadata")
        if args.seed_info:
            print("\nSeed History:")
            for entry in seed_manager.seed_history:
                print(f"Seed: {entry['seed']}")
                print(f"Time: {entry['timestamp']}")
                print(f"Auto-generated: {entry['generated']}\n")
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()