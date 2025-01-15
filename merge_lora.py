import os
import torch
from safetensors.torch import load_file, save_file
from collections import defaultdict
from tqdm import tqdm
import gc
import math
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import time
from pathlib import Path

@dataclass
class MergeState:
    """Tracks the state of the merge process."""
    processed_keys: set
    modified_counts: Dict[str, int]
    last_checkpoint: str
    total_modifications: int
    start_time: float

    def to_dict(self) -> Dict:
        return {
            "processed_keys": list(self.processed_keys),
            "modified_counts": self.modified_counts,
            "last_checkpoint": self.last_checkpoint,
            "total_modifications": self.total_modifications,
            "elapsed_time": time.time() - self.start_time
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MergeState':
        return cls(
            processed_keys=set(data.get("processed_keys", [])),
            modified_counts=defaultdict(int, data.get("modified_counts", {})),
            last_checkpoint=data.get("last_checkpoint", ""),
            total_modifications=data.get("total_modifications", 0),
            start_time=time.time() - data.get("elapsed_time", 0)
        )

class LoRAMerger:
    def __init__(self, base_model_path: str, output_path: str, device: str = None):
        self.logger = self._setup_logging()
        self.base_model_path = Path(base_model_path)
        self.output_path = Path(output_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_file = self.output_path.with_suffix('.state.json')
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging with more detailed formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("lora_merger.log")
            ]
        )
        return logging.getLogger("LoRA Merger")

    def _clear_memory(self, aggressive: bool = False):
        """Enhanced memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()

    def _load_state_dict(self, file_path: str, retries: int = 3) -> Dict:
        """Load state dict with retry mechanism."""
        for attempt in range(retries):
            try:
                self._clear_memory()
                self.logger.info(f"Loading {Path(file_path).name}")
                if str(file_path).endswith('.safetensors'):
                    return load_file(file_path)
                return torch.load(file_path, map_location='cpu')
            except Exception as e:
                if attempt < retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(1)
                else:
                    self.logger.error(f"Failed to load {file_path} after {retries} attempts: {e}")
                    raise

    def _save_checkpoint(self, current_dict: Dict, state: MergeState):
        """Save checkpoint with state tracking."""
        temp_output = self.output_path.with_suffix('.temp')
        temp_state = self.state_file.with_suffix('.temp.json')
        
        try:
            # Save model checkpoint
            if str(self.output_path).endswith('.safetensors'):
                save_file(current_dict, str(temp_output))
            else:
                torch.save(current_dict, temp_output)

            # Save state
            with open(temp_state, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename
            temp_output.replace(self.output_path)
            temp_state.replace(self.state_file)
            
            self.logger.info(f"Checkpoint saved: {self.output_path.name}")
            self._clear_memory(aggressive=True)
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
            if temp_output.exists():
                temp_output.unlink()
            if temp_state.exists():
                temp_state.unlink()
            raise

    def _process_tensor(self, base_tensor: torch.Tensor, 
                       lora_down: torch.Tensor, 
                       lora_up: torch.Tensor, 
                       scale: float) -> torch.Tensor:
        """Process tensor with error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if len(base_tensor.shape) == 2:
                    contribution = torch.mm(lora_up, lora_down) * scale
                else:
                    contribution = torch.nn.functional.conv2d(
                        lora_down.permute(1, 0, 2, 3),
                        lora_up
                    ).permute(1, 0, 2, 3) * scale

                if contribution.shape != base_tensor.shape:
                    raise ValueError(f"Shape mismatch: {contribution.shape} vs {base_tensor.shape}")

                return base_tensor + contribution

            except torch.cuda.OutOfMemoryError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"OOM on attempt {attempt + 1}, clearing memory and retrying...")
                    self._clear_memory(aggressive=True)
                    # Move tensors to CPU, then back to GPU
                    base_tensor = base_tensor.cpu()
                    lora_down = lora_down.cpu()
                    lora_up = lora_up.cpu()
                    time.sleep(1)  # Give system time to recover
                    base_tensor = base_tensor.to(self.device)
                    lora_down = lora_down.to(self.device)
                    lora_up = lora_up.to(self.device)
                else:
                    raise

    def merge_loras(self,
                   lora_configs: List[Tuple[str, float]],
                   save_every: int = 500,
                   boost_factor: float = 1.5):
        """Enhanced LoRA merging with state tracking and error recovery."""
        
        # Load or create merge state
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = MergeState.from_dict(json.load(f))
            self.logger.info(f"Resuming merge from checkpoint with {len(state.processed_keys)} processed keys")
            merged_dict = self._load_state_dict(str(self.output_path))
        else:
            state = MergeState(set(), defaultdict(int), "", 0, time.time())
            merged_dict = self._load_state_dict(str(self.base_model_path))

        # Get model properties
        model_keys = list(merged_dict.keys())
        sample_tensor = next(tensor for tensor in merged_dict.values() if isinstance(tensor, torch.Tensor))
        original_dtype = sample_tensor.dtype
        self.logger.info(f"Model precision: {original_dtype}")

        # Load and prepare LoRA states
        lora_states = []
        for lora_path, ratio in lora_configs:
            try:
                lora_sd = self._load_state_dict(lora_path)
                
                # Extract and validate LoRA parameters
                alphas = {}
                dims = {}
                for key, value in lora_sd.items():
                    if isinstance(value, torch.Tensor):
                        if value.dtype != original_dtype:
                            lora_sd[key] = value.to(original_dtype)
                    
                    if 'alpha' in key:
                        module_name = key.split('.alpha')[0]
                        alphas[module_name] = value.item()
                    elif 'lora_down' in key:
                        module_name = key.split('.lora_down')[0]
                        dims[module_name] = value.size(0)

                # Set default alphas
                for module_name in dims:
                    alphas.setdefault(module_name, dims[module_name])

                lora_states.append((lora_sd, ratio * boost_factor, alphas))
                self.logger.info(f"Loaded LoRA from {Path(lora_path).name}: {len(dims)} modules")
                
            except Exception as e:
                self.logger.error(f"Failed to load LoRA {lora_path}: {e}")
                raise

        # Process in batches
        current_batch = {}
        pbar = tqdm(model_keys, desc="Processing layers")
        
        for key in pbar:
            if key in state.processed_keys:
                continue

            try:
                current_batch[key] = merged_dict[key].clone().to(self.device)

                # Apply LoRA modifications
                for lora_sd, ratio, alphas in lora_states:
                    for lora_key, lora_value in lora_sd.items():
                        if 'lora_down' not in lora_key:
                            continue

                        module_name = lora_key.split('.lora_down')[0]
                        if key not in module_name:
                            continue

                        try:
                            down_weight = lora_value.to(self.device)
                            up_weight = lora_sd[lora_key.replace('lora_down', 'lora_up')].to(self.device)
                            
                            alpha = alphas.get(module_name, down_weight.size(0))
                            scale = math.sqrt(alpha) * ratio

                            current_batch[key] = self._process_tensor(
                                current_batch[key], down_weight, up_weight, scale
                            )

                            state.modified_counts[module_name] += 1
                            state.total_modifications += 1

                            del down_weight, up_weight
                            
                        except Exception as e:
                            self.logger.error(f"Error processing {key} for {module_name}: {e}")
                            raise

                # Move processed tensor back to CPU
                current_batch[key] = current_batch[key].cpu()
                state.processed_keys.add(key)

                # Update progress
                pbar.set_postfix({
                    'Modified': state.total_modifications,
                    'Memory': f"{torch.cuda.max_memory_allocated() // 1024**2}MB" if torch.cuda.is_available() else "N/A"
                })

                # Save checkpoint at intervals
                if len(current_batch) >= save_every:
                    merged_dict.update(current_batch)
                    self._save_checkpoint(merged_dict, state)
                    current_batch = {}

            except Exception as e:
                self.logger.error(f"Error processing {key}: {e}")
                self._clear_memory(aggressive=True)
                continue

        # Save final results
        if current_batch:
            merged_dict.update(current_batch)
        self._save_checkpoint(merged_dict, state)

        # Final statistics
        elapsed_time = time.time() - state.start_time
        self.logger.info(f"Merge completed in {elapsed_time:.2f} seconds")
        for module_name, count in state.modified_counts.items():
            self.logger.info(f"Modified {count} layers for {module_name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced LoRA Merger")
    parser.add_argument('--base_model', required=True, help="Path to base model")
    parser.add_argument('--lora_paths', nargs='+', required=True, help="Paths to LoRA models")
    parser.add_argument('--ratios', nargs='+', type=float, required=True, help="Merge ratios")
    parser.add_argument('--output_path', required=True, help="Output path")
    parser.add_argument('--save_every', type=int, default=500, help="Save checkpoint every N layers")
    parser.add_argument('--boost_factor', type=float, default=1.5, help="LoRA contribution boost factor")
    parser.add_argument('--device', help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    if len(args.lora_paths) != len(args.ratios):
        raise ValueError("Number of LoRA paths must match number of ratios!")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Initialize and run merger
    merger = LoRAMerger(args.base_model, args.output_path, args.device)
    merger.merge_loras(
        list(zip(args.lora_paths, args.ratios)),
        args.save_every,
        args.boost_factor
    )
