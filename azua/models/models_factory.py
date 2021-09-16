import os
from typing import Union, Dict, Any, Type
from uuid import uuid4

from ..datasets.variables import Variables
from ..models.model import Model
from ..models.point_net import PointNet, SparsePointNet
from ..models.set_encoder_base_model import SetEncoderBaseModel
from ..models.transformer_set_encoder import TransformerSetEncoder
from ..utils.factory_utils import get_named_subclass


def create_model(
    model_name: str,
    models_dir: str,
    variables: Variables,
    device: Union[str, int],
    model_config_dict: Dict[str, Any],
    model_id: str = None,
) -> Model:
    """
    Get an instance of a concrete implementation of the `Model` class.

    Args:
        model_name (str): String corresponding to concrete instance of `Objective` class.
        models_dir (str): Directory to save model information in.
        variables (Variables): Information about variables/features used
                by this model.
        model_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
            the form {arg_name: arg_value}. e.g. {"sample_count": 10}
        device (str or int): Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
            (e.g. 0 or 1 on a two-GPU machine).
        model_id (str): String specifying GUID for model. A GUID will be generated if not provided.

    Returns:
        Instance of concrete implementation of `Model` class.
    """
    # Create anything needed for all model types.
    model_id = model_id if model_id is not None else str(uuid4())
    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir)

    model_class = get_named_subclass(["models", "baselines"], Model, model_name)

    return model_class.create(model_id, save_dir, variables, model_config_dict, device=device)


def load_model(model_id: str, models_dir: str, device: Union[str, int]) -> Model:
    """
    Loads an instance of a concrete implementation of the `Model` class.

    Args:
        model_id (str): String corresponding to model's id.
        models_dir (str): Directory where mnodel information is saved.

    Returns:
        Deseralized instance of concrete implementation of `Model` class.
    """
    model_type_filepath = os.path.join(models_dir, "model_type.txt")
    with open(model_type_filepath) as f:
        model_name = f.read()

    model_class = get_named_subclass(["models", "baselines"], Model, model_name)

    return model_class.load(model_id, models_dir, device)


def create_set_encoder(set_encoder_type: str, kwargs: dict) -> SetEncoderBaseModel:
    """
    Create a set encoder instance.

    Args:
        set_encoder_type (str): type of set encoder to create
        kwargs (dict): keyword arguments to pass to the set encoder constructor

    """
    # Create a set encoder instance.
    set_encoder_type_map: Dict[str, Type[SetEncoderBaseModel]] = {
        "default": PointNet,
        "sparse": SparsePointNet,
        "transformer": TransformerSetEncoder,
    }
    return set_encoder_type_map[set_encoder_type](**kwargs)
