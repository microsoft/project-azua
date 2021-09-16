import inspect
import itertools
from typing import Any, Dict, Union, Optional

import torch
import torch.nn as nn


from .nn.mixins.base import BayesianMixin


def _subclasses(cls: type):
    return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in _subclasses(c)])


_BAYESIANIZABLE_CLASSES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
# dictionary mapping nn.Module classes for which we support 'Bayesianization' to a dictionary mapping
# an inference name (the base name of the mixin class) to a class inheriting from the nn.Module and the
# inference Mixin. For example, nn.Linear would map to a dictionary containing `'ffg'=FFGLinear` as one
# of the key-value pairs. The dictionary construction assumes that implementations of Bayesian layers
# inherit from the nn.Module as the final baseclass, so that an arbitrary number of additional mixin
# classes is supported
_BAYESIAN_MODULES = {
    t_cls: {
        m_cls.__bases__[0].__name__.rstrip("Mixin").lower(): m_cls
        for m_cls in _subclasses(BayesianMixin)
        if len(m_cls.__bases__) >= 2 and m_cls.__bases__[-1] == t_cls
    }
    for t_cls in _BAYESIANIZABLE_CLASSES
}


def _deep_setattr(obj: Any, name: str, value: Any) -> None:
    attr_names = name.split(".")
    for attr_name in attr_names[:-1]:
        obj = getattr(obj, attr_name)
    setattr(obj, attr_names[-1], value)


def _module_in_sd(module_name: str, sd: Dict):
    return any(k.startswith(module_name) for k in sd.keys())


def bayesian_from_template(layer: nn.Module, inference: str, **bayes_kwargs) -> BayesianMixin:
    """Takes a pytorch module and turns it into an equivalent Bayesian module depending on inference.
    For example, if layer is an nn.Linear instance and inference is ffg, the method constructs an
    FFGLinear object with the same number of in_features and out_features."""
    bayes_module_class = _BAYESIAN_MODULES[layer.__class__][inference.lower()]
    # pulls nn.Module init arguments from the attributes of the object. bias needs to be treated separately
    # since init expects a bool, while it is a torch.tensor or None as an attribute
    init_parameters = inspect.signature(layer.__class__).parameters
    layer_kwargs = {k: v for k, v in vars(layer).items() if k in init_parameters}
    layer_kwargs["bias"] = getattr(layer, "bias", None) is not None
    return bayes_module_class(**layer_kwargs, **bayes_kwargs)


def bayesianize_(
    network: nn.Module,
    inference: Union[str, Dict[Union[str, nn.Module, type, int], Any]],
    reference_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    **default_params
) -> None:
    """Method for turning a pytorch neural network that is an instance of nn.Module into a 'Bayesian'
    variant, where all of the nn.Linear and nn.ConvNd layers are replaced (inplace) with Bayesian layers.
    Which type of Bayesian layer gets used is specified by inference. If it is a string, the same type of
    layer is used throughout the net, otherwise a dictionary mapping specific layer names or objects or
    entire classes or the module's index to a string can be used. That way it is possible to, for example,
    only do variational inference on the output layer and learn the parameters of the remaining layers
    via maximum likelihood."""
    if reference_state_dict is None:
        reference_state_dict = {}

    num_modules = len(list(network.modules()))
    for i, (name, module) in enumerate(network.named_modules()):
        if isinstance(inference, str):
            module_inference: Any = inference
        elif module in inference:
            module_inference = inference[module]
        elif name in inference:
            module_inference = inference[name]
        elif i in inference:
            module_inference = inference[i]
        elif i - num_modules in inference:
            module_inference = inference[i - num_modules]
        elif module.__class__ in inference:
            module_inference = inference[module.__class__]
        elif module.__class__.__name__ in inference:
            module_inference = inference[module.__class__.__name__]
        else:
            continue

        if isinstance(module_inference, str):
            module_inference = {"inference": module_inference}

        for k, v in default_params.items():
            module_inference.setdefault(k, v)

        cls = module.__class__
        if cls in _BAYESIAN_MODULES and module_inference["inference"] in _BAYESIAN_MODULES[cls]:
            bayesian_module = bayesian_from_template(module, **module_inference)
            _deep_setattr(network, name, bayesian_module)

            if _module_in_sd(name, reference_state_dict):
                param_dict = {
                    param_name: reference_state_dict[name + "." + param_name]
                    for param_name, _ in module.named_parameters()
                }
                bayesian_module.init_from_deterministic_params(param_dict)
        else:
            if _module_in_sd(name, reference_state_dict):
                module_sd = {
                    attr_name: reference_state_dict[name + "." + attr_name]
                    for attr_name, _ in itertools.chain(
                        module.named_parameters(recurse=False), module.named_buffers(recurse=False),
                    )
                }
                # check for non-empty state dict, some modules, e.g. BasicBlock in Resnet, only contain other
                # modules and don't have any parameters of their own
                if module_sd:
                    module.load_state_dict(module_sd)
