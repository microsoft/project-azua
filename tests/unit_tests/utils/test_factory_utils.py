from azua.models.model import Model
from azua.utils.factory_utils import get_subclasses, get_named_subclass
from azua.models.pvae_base_model import PVAEBaseModel


def test_get_named_subclass():
    subclass = get_named_subclass(["azua/models"], Model, "pvae")
    assert subclass.name() == "pvae"


def test_get_subclasses():

    subclass_list = get_subclasses("azua/models", PVAEBaseModel)

    subclass_set_names = {c.name() for c in subclass_list}
    expected_subclasses = {"pvae", "visl", "vaem", "transformer_pvae", "nri"}
    assert expected_subclasses <= subclass_set_names
