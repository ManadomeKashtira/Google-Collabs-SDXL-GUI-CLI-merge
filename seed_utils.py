import random
import torch
import numpy as np
import hashlib
import time
from datetime import datetime

class SeedManager:
    def __init__(self):
        self.current_seed = None
        self.seed_history = []
        
    def generate_seed(self):
        """Generate a deterministic but random-appearing seed"""
        timestamp = int(time.time() * 1000)
        hash_input = f"{timestamp}{random.getrandbits(32)}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        return int(hash_value[:8], 16)  # Use first 8 chars for a 32-bit integer
        
    def set_seed(self, seed=None, track=True):
        """Set or generate seed and track it"""
        if seed is None:
            seed = self.generate_seed()
            
        self.current_seed = seed
        
        # Set all relevant seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        if track:
            self.seed_history.append({
                'seed': seed,
                'timestamp': datetime.now().isoformat(),
                'generated': seed is None
            })
            
        return seed
        
    def get_seed_info(self):
        """Get current seed information"""
        return {
            'current_seed': self.current_seed,
            'history': self.seed_history
        }
        
    def save_seed_info(self, metadata):
        """Save seed information to metadata"""
        metadata['seed_info'] = self.get_seed_info()
        return metadata