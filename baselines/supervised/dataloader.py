from torch.utils.data import DataLoader

from dataset import BaselineDataset


class BaselineDataLoader(DataLoader):
    def __init__(self,
                 dataset: BaselineDataset,
                 batch_size: int,
                 shuffle: bool = False,
                 num_workers: int = 0
                 ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate
        )
