import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob

from dgl.geometry import farthest_point_sampler

# distributed sampler
from torch.utils.data.distributed import DistributedSampler


def load_dataset(args, mode):
    if args.data.dataset == "shapenet":
        batch_size = args.training.batch_size if mode != 'val' else 32
        dataset = ShapeNet(
            root=args.data.dataset_dir,
            mode=mode,
            num_samples=args.data.num_samples
        )
        datasampler = DistributedSampler(
            dataset=dataset
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=datasampler,
            num_workers=args.data.num_workers,
            pin_memory=True
        )
        return dataset, dataloader
    
    else:
        raise Exception(f'Unsupported dataset: {args.data.dataset}')


# iterable dataset
class ShapeNet(Dataset):
    def __init__(self, root, mode, num_samples):
        super().__init__()
        self.root = root
        self.num_samples = num_samples

        self.directories = []
        for dir in glob.glob(f"{root}/*"):
            with open(f"{dir}/{mode}.lst") as file:
                for line in file:
                    self.directories.append(f"{dir}/{line.strip()}")

    def __len__(self):
        """return directories size"""
        return len(self.directories)
    
    def __getitem__(self, index):
        """return specify object, point cloud & image"""
        dir = self.directories[index]
        data = np.load(f"{dir}/points.npz")
        positions = data["points"].astype(np.float32)
        occupancies = np.unpackbits(data["occupancies"])[:positions.shape[0]].astype(np.float32)

        # at least half postive samples
        half_samples = self.num_samples // 2

        # near surface region samples
        near_surface_indices = np.nonzero(occupancies.astype(np.int32))[0]
        if near_surface_indices.shape[0] <= half_samples:
            indices_of_near_surface_indices = np.random.choice(near_surface_indices.shape[0], half_samples)
        else:
            surface_positions = torch.tensor(positions[near_surface_indices]).unsqueeze(0)
            indices_of_near_surface_indices = farthest_point_sampler(surface_positions, half_samples).numpy()[0]
                
        near_surface_indices = near_surface_indices[indices_of_near_surface_indices]

        # random samples
        random_indices = np.random.choice(positions.shape[0], half_samples, replace=False)

        # integrate indices
        indices = np.concatenate([near_surface_indices, random_indices], axis=0)
        indices = np.random.permutation(indices)

        surfaces = positions[near_surface_indices]
        queries = positions[indices]
        occupancies = occupancies[indices]

        return surfaces, queries, occupancies
