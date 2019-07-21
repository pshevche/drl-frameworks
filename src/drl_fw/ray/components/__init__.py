from ray.rllib.models import ModelCatalog
from drl_fw.ray.components.parametric_model import ParametricActionsModel

ModelCatalog.register_custom_model(
    "parametric_actions_model", ParametricActionsModel)
