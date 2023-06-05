import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, io
import torch
from glob import glob


class SR2DDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 1

    # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
    def setup(self, stage: str) -> None:
        allset = SR2DDataset()
        self.train_set, self.val_set, self.test_set = random_split(allset, [0.6, 0.2, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)


class SR2DDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.files = glob('data/soilct/*/*')
        self.transform_x = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                               transforms.CenterCrop(256),
                                               transforms.Resize(128, antialias=True)])
        self.transform_y = transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                               transforms.CenterCrop(256),
                                               ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return {
            'name': self.files[index],
            'putin': self.transform_x(io.read_image(self.files[index], io.ImageReadMode.GRAY)),
            'target': self.transform_y(io.read_image(self.files[index], io.ImageReadMode.GRAY))
        }
