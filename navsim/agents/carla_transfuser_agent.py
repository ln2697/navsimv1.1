import json
import os
from tabnanny import check
from typing import List, Dict, Union

import cv2
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
from open_loop_inference import OpenLoopPrediction as CarlaOpenLoopPrediction
from constants import SourceDataset as CarlaSourceDataset
import numpy as np

class CarlaTransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""
    def __init__(
        self,
        config: TransfuserConfig,
        checkpoint_path: str
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent.
        :param checkpoint_path: optional path string to checkpoint, defaults to None.
        """
        super().__init__()

        self._config = config

        self._checkpoint_path = checkpoint_path.rsplit("/", 1)[0]
        model_filename = checkpoint_path.rsplit("/", 1)[1]
        print(self._checkpoint_path, model_filename)
        
        with open(os.path.join(self._checkpoint_path, "config.json"), "r", encoding="utf-8") as f:
            json_config = json.load(f)
        self._carla_model_config = CarlaTrainingConfig(json_config)
        self._carla_config_open_loop = CarlaOpenLoopConfig()
        self._carla_open_loop_inference = CarlaOpenLoopInference(
            config_training=self._carla_model_config,
            config_open_loop=self._carla_config_open_loop,
            model_path=self._checkpoint_path,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            prefix=model_filename
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
        
        rgb = cv2.imdecode(np.frombuffer(features["camera_feature"], np.uint8), cv2.IMREAD_COLOR)
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC to CHW
        rgb = torch.tensor(rgb).unsqueeze(0).float()  # CHW to NCHW
        print(rgb.shape, features["status_feature"].shape)
        output: CarlaOpenLoopPrediction = self._carla_open_loop_inference({
            "rgb": rgb,
            "command": features["status_feature"][:4].reshape(-1, 4),
            "speed": torch.linalg.norm(features["status_feature"][4:6]).reshape(-1, 1),
            "acceleration": torch.linalg.norm(features["status_feature"][6:8]).reshape(-1, 1),
        })
        
        return {
            "trajectory": torch.concatenate([output.pred_future_waypoints, output.pred_future_headings.unsqueeze(-1)], dim=-1)
        }

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
