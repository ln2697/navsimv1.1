import json
import os
from typing import List, Dict, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from config_open_loop import OpenLoopConfig as CarlaOpenLoopConfig
from config_training import TrainingConfig as CarlaTrainingConfig
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from open_loop_inference import OpenLoopInference as CarlaOpenLoopInference

class CarlaTransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: str,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent.
        :param checkpoint_path: optional path string to checkpoint, defaults to None.
        """
        super().__init__()

        self._config = config

        self._checkpoint_path = checkpoint_path
        
        with open(os.path.join(self._checkpoint_path, "config.json"), "r", encoding="utf-8") as f:
            json_config = json.load(f)
        self._carla_model_config = CarlaTrainingConfig(json_config)
        self._carla_config_open_loop = CarlaOpenLoopConfig()
        self._carla_open_loop_inference = CarlaOpenLoopInference(
            config_training=self._carla_model_config,
            config_open_loop=self._carla_config_open_loop,
            model_path=self._checkpoint_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            prefix="model"
        )

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        raise NotImplementedError("CARLA TransFuser supports only inference.")
