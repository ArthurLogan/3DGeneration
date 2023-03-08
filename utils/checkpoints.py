import os
import torch


class CheckpointIO:
    def __init__(self, checkpoint_dir='./ckpt', **kwargs):
        self.module_dict = kwargs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def register_modules(self, **kwargs):
        """registers modules in current module dictionary."""
        self.module_dict.update(kwargs)
    
    def save(self, filename, **kwargs):
        """saves the current module dictionary."""
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        
        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)
    
    def load(self, filename):
        """loads a module dictionary from local file"""
        if not os.path.isabs(filename):
            filename = os.path.join(self.checkpoint_dir, filename)
        
        if os.path.exists(filename):
            print(filename)
            print('=> Loading checkpoint from local file...')
            state_dict = torch.load(filename)
            scalars = self.parse_state_dict(state_dict)
            return scalars
        else:
            return FileExistsError
    
    def parse_state_dict(self, state_dict):
        """parse state_dict of model and return scalars"""
        for k, v in self.module_dict.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                print('Warning: Could not find %s in checkpoint!' % k)
        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
        return scalars
