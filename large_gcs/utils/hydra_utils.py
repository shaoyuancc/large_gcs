import importlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_function_from_string(function_string):
    """Get function from string.

    Args:
        function_string (str): String representation of function.
    Returns:
        function: Function.
    """
    module_name, function_name = function_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


def get_cfg_from_folder(path: Path) -> DictConfig:
    # Load the config file for this run
    config_path = path / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"The config file {config_path} does not exist.")
    cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore
    return cfg
