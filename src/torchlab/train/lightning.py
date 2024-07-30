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
# the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.profilers import Profiler

from torchlab.utils import load_config


class LabTrainer(pl.Trainer):
    def __init__(
        self,
        cfg_path: Path,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = None,
        callbacks: Optional[List] = None,
        plugins: Optional[List] = None,
        set_seed: bool = True,
        **kwargs: Dict[str, Any]
    ) -> None:
        cfg = load_config(cfg_path)
        if set_seed:
            seed_everything(cfg.seed, workers=True)
        if not callbacks:
            callbacks = []
        if not plugins:
            plugins = []
        super().__init__(
            logger=logger,
            profiler=profiler,
            callbacks=callbacks + [ModelCheckpoint(dirpath=cfg.checkpoints.experiments, filename="model")],
            plugins=plugins,
            **kwargs
        )
