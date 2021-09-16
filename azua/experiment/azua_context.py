from typing import Any, Callable, Dict, List, Optional
from .imetrics_logger import IMetricsLogger, ISystemMetricsLogger
from dependency_injector import containers, providers


class AzuaContext(containers.DeclarativeContainer):
    def mock_download_dataset(dataset_name: str, data_dir: str):
        raise NotImplementedError("No download_dataset functionality provided")

    def is_azureml_run_local_run():
        return False

    class MockMetricLogger(IMetricsLogger):
        def log_value(self, metric_name: str, value: Any, log_to_parent: Optional[bool] = False):
            pass

        def log_list(self, metric_name: str, values: List[Any], log_to_parent: Optional[bool] = False):
            pass

        def set_tags(self, tags: Dict[str, Any], log_to_parent: Optional[bool] = False):
            pass

        def finalize(self):
            pass

        def log_dict(self, metrics: Dict[str, Any], log_to_parent: Optional[bool] = False):
            pass

    class MockSystemMetricsLogger(ISystemMetricsLogger):
        def start_log(self):
            pass

        def end_log(self):
            pass

    def aml_step(func: Callable, creation_mode: bool) -> Callable:
        return func

    download_dataset = providers.Callable(mock_download_dataset)
    is_azureml_run = providers.Callable(is_azureml_run_local_run)
    metrics_logger = providers.Object(MockMetricLogger())
    system_metrics_logger = providers.Object(MockSystemMetricsLogger())
    pipeline = providers.Object(None)  # TODO: Add IEvaluationPipeline to azua/?
    aml_step = providers.Callable(aml_step)
