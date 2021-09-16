class ModelClassNotFound(NotImplementedError):
    pass


class DatasetNotFound(FileNotFoundError):
    pass


class ONNXNotImplemented(NotImplementedError):
    pass


class ModelConfigNotFound(Exception):
    pass


class ValidationDataNotAvailable(Exception):
    # Raise this when you try to do something that requires validation data, but there is no validation data
    pass
