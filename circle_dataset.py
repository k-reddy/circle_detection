import itertools
from torch import from_numpy, tensor, stack, Tensor
from torch.utils.data import Dataset

from circle_detection import generate_examples


class CircleDataset(Dataset):
    def __init__(self, num_samples, noise_level):
        raw_data = list(
            itertools.islice(generate_examples(noise_level=noise_level), num_samples)
        )

        self.images = stack([from_numpy(d[0]).float().unsqueeze(0) for d in raw_data])
        self.targets = stack(
            [tensor([d[1].row, d[1].col, d[1].radius]).float() for d in raw_data]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
