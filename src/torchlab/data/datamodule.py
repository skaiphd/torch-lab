# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, Dataset

from torchlab.data.dataset import LabDataSet


class LabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset = LabDataSet,
        data_dir: str = "data",
        split: bool = True,
        train_size: float = 0.8,
        num_workers: int = mp.cpu_count() // 2,
        transforms=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset = dataset
        self.split = split
        self.train_size = train_size
        self.num_workers = num_workers
        self.transforms = transforms

    def prepare_data(self):
        self.dataset(self.data_dir)

    def setup(self, stage=None):
        if stage == "fit":
            full_dataset = self.dataset(self.data_dir, train=True, transform=self.transforms)
            train_size = int(len(full_dataset) * self.train_size)
            test_size = len(full_dataset) - train_size
            self.train_data, self.val_data = random_split(full_dataset, lengths=[train_size, test_size])
        if stage == "test":
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers)
