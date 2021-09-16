from typing import Dict, Any, Union

from ..models.imodel import IModelForObjective
from ..models.transformer_imputer import TransformerImputer
from ..objectives.objective import Objective
from ..utils.factory_utils import get_named_subclass


def create_objective(
    objective_name: str, model: Union[IModelForObjective, TransformerImputer], obj_config_dict: Dict[str, Any]
) -> Objective:
    """
    Get an instance of a concrete implementation of the `Objective` class.

    Args:
        objective_name (str): String corresponding to concrete instance of `Objective` class.
        model (Model): Trained `Model` class to use.
        obj_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
            the form {arg_name: arg_value}. e.g. {"sample_count": 10}

    Returns:
        Instance of concrete implementation of `Objective` class.
    """
    obj_class = get_named_subclass("objectives", Objective, objective_name)
    return obj_class(model, **obj_config_dict)
