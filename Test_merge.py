import os
import json
import torch
import random
from datetime import datetime
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc
import numpy as np

class ModelMerger:
    MERGE_METHODS = {
        "weighted_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)),
        "sigmoid": lambda tensors, ratios: torch.sigmoid(sum(t * r for t, r in zip(tensors[1:], ratios[1:]))) * tensors[0],
        "geometric": lambda tensors, ratios: torch.prod(torch.stack([torch.pow(t, r) for t, r in zip(tensors, ratios)]), dim=0),
        "max": lambda tensors, ratios: torch.max(torch.stack([t * r for t, r in zip(tensors, ratios)]), dim=0)[0],
        "add_difference": lambda tensors, ratios: tensors[0] + sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "smooth_add_difference": lambda tensors, ratios: tensors[0] + torch.tanh(sum((t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "multiply_difference": lambda tensors, ratios: tensors[0] * (1 + sum((t/tensors[0] - 1) * r for t, r in zip(tensors[1:], ratios[1:]))),
        "similarity": lambda tensors, ratios: (sum(r * torch.cosine_similarity(tensors[0], t, dim=0).unsqueeze(-1) * t for t, r in zip(tensors, ratios))) / sum(ratios),
        "train_difference": lambda tensors, ratios: tensors[0] + sum(torch.sign(t - tensors[0]) * r for t, r in zip(tensors[1:], ratios[1:])),
        "triple_sum": lambda tensors, ratios: sum(t * r for t, r in zip(tensors, ratios)) / torch.norm(tensors[0]),
        "tensor_sum": lambda tensors, ratios: torch.stack([t * r for t, r in zip(tensors, ratios)]).sum(0),
        "sum_twice": lambda tensors, ratios: sum(tensors[0] + t * r for t, r in zip(tensors[1:], ratios[1:])) / len(tensors[1:])
    }

    PRECISION_TYPES = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "prunefp16": "prunefp16"  # Special handling for pruned FP16
    }

    def __init__(self, config):
        self.config = config
        self.merge_method = self.MERGE_METHODS[config["merge_method"]]
        self.precision = self.PRECISION_TYPES[config["precision"]]
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.chunk_size = 100  # Reduced for better memory management
        random.seed(config["merge_seed"])
        
        self.num_models = len(config["model_paths"])
        if self.num_models == 2:
            self.ratios = [1 - config["alpha"], config["alpha"]]
        else:
            self.ratios = [1 - config["alpha"] - config["beta"], config["alpha"], config["beta"]]

    def get_model_size(self, model_dict):
        total_size = 0
        for tensor in model_dict.values():
            total_size += tensor.nelement() * tensor.element_size()
        return total_size / (1024 ** 3)  # Convert to GB
        
    def get_component_type(self, key):
        if key.startswith('model.diffusion_model'):
            return 'UNET'
        elif key.startswith('first_stage_model'):
            return 'VAE'
        elif key.startswith('transformer_'):
            return 'TEXT_ENCODER'
        else:
            return 'OTHER'
            
    def load_checkpoint(self, path):
        print(f"Loading checkpoint: {path}")
        model = load_file(path)
        size = self.get_model_size(model)
        print(f"Model size: {size:.2f} GB")
        return model

    def merge_tensors(self, tensors, component_type):
        # Handle different precisions
        if self.precision == "prunefp16":
            dtype = torch.float16
        else:
            dtype = self.precision

        if component_type == 'VAE':
            vae_idx = {"first": 0, "second": 1, "last": -1}[self.config["vae_source"]]
            return tensors[vae_idx].to(dtype)

        try:
            tensors = [t.to(self.device, dtype) for t in tensors]
            merged = self.merge_method(tensors, self.ratios)
            
            # Ensure we maintain tensor properties
            merged = merged.to(dtype)
            if hasattr(tensors[0], 'stride'):
                merged = merged.contiguous()
            
            return merged.cpu()
        except Exception as e:
            print(f"Error merging tensor: {str(e)}")
            return tensors[0].to(dtype)

    def merge(self):
        try:
            models = [self.load_checkpoint(path) for path in self.config["model_paths"]]
            all_keys = set(models[0].keys())
            merged_model = {}
            
            print(f"\nMerging {len(self.config['model_paths'])} models...")
            print(f"Merge method: {self.config['merge_method']}")
            print(f"Ratios: {self.ratios}")
            
            # Track sizes for verification
            input_sizes = [self.get_model_size(model) for model in models]
            print(f"Input model sizes: {[f'{size:.2f} GB' for size in input_sizes]}")

            # Process in smaller chunks for memory efficiency
            key_list = list(all_keys)
            for i in tqdm(range(0, len(key_list), self.chunk_size), desc="Merging chunks"):
                chunk_keys = key_list[i:i + self.chunk_size]
                chunk_dict = {}
                
                for key in chunk_keys:
                    if key not in models[0]:
                        continue

                    component_type = self.get_component_type(key)
                    tensors = [m.get(key, torch.zeros_like(models[0][key])) for m in models]
                    
                    merged_tensor = self.merge_tensors(tensors, component_type)
                    chunk_dict[key] = merged_tensor

                # Save chunk
                if i == 0:
                    save_file(chunk_dict, self.config["output_path"])
                else:
                    existing = load_file(self.config["output_path"])
                    existing.update(chunk_dict)
                    save_file(existing, self.config["output_path"])

                gc.collect()
                torch.cuda.empty_cache()

            # Verify final size
            final_model = load_file(self.config["output_path"])
            final_size = self.get_model_size(final_model)
            print(f"\nFinal model size: {final_size:.2f} GB")
            
            if final_size < min(input_sizes) * 0.8:  # Size sanity check
                print("Warning: Final model size is significantly smaller than inputs!")
                print("This might indicate a merging issue.")
            
            return True
            
        except Exception as e:
            print(f"Error during merge: {str(e)}")
            return False