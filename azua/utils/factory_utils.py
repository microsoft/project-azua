import logging
import os
from importlib import import_module
from typing import Any, List, Type, Union


logger = logging.getLogger(__name__)


class ModelClassNotFound(NotImplementedError):
    pass


def get_subclasses(directory_path: str, parent_class: Type[Any]):
    subclass_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), directory_path
    )

    for root, dirs, files in os.walk(subclass_dir):
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")
        for subclass_file in files:
            subclass_path = os.path.relpath(os.path.join(root, subclass_file), subclass_dir)
            subclass_file_name, ext = os.path.splitext(subclass_path)
            if ext == ".py" and not subclass_file.endswith("__init__"):
                # TODO: change it to the relative import. This way,
                # when we opensource the code, we shouldn't need to
                # change this line
                package = directory_path.strip("/").replace("/", ".")
                module = subclass_file_name.replace("/", ".")
                subclass_import_path = f"{package}.{module}"
                import_module(subclass_import_path)
    subclass_list = parent_class.__subclasses__()
    return subclass_list


def get_named_subclass(
    directories: Union[str, List[str]], parent_class: Type[Any], subclass_name: str, max_depth: int = 10
) -> Type[Any]:
    """
    Get a named subclass of parent_class from directory.
    Args:
        directories: path to directory or list of paths to directories to search for subclass. The function will iterate
            through this list first to last and return the first time a subclass of matching name is found.
        parent_class: class to find named subclass of.
        subclass_name: name of subclass to search for.
        max_depth: maximum number of subclass levels to search.
    Returns:
        subclass: subclass of `parent_class` with the name `subclass_name`, if such a subclass exists.
    """
    if isinstance(directories, str):
        directories = [directories]

    for directory in directories:
        subclass_list = get_subclasses(directory, parent_class)

        for _ in range(max_depth):
            children = [
                subsubclass for subclass in subclass_list for subsubclass in get_subclasses(directory, subclass)
            ]
            if set(children).issubset(set(subclass_list)):
                break
            else:
                subclass_list.extend(children)

        while subclass_list:
            subclass = subclass_list.pop()

            # If class is abstract, get its subclasses instead.
            if hasattr(subclass, "__abstractmethods__") and len(subclass.__abstractmethods__) > 0:
                subclass_list.extend(subclass.__subclasses__())
                continue

            if subclass.name() == subclass_name:
                return subclass

    raise ModelClassNotFound("Subclass with name %s not found." % subclass_name)
