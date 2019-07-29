from ray.rllib.models import ModelCatalog
from drl_fw.ray.components.parametric_models import (
    ParametricDQNModel,
    ParametricRainbowModel)

ModelCatalog.register_custom_model(
    "parametric_dqn_model", ParametricDQNModel)

ModelCatalog.register_custom_model(
    "parametric_rainbow_model", ParametricRainbowModel)
