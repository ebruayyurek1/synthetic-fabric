from pathlib import Path

from src.utils.io_utils import load_yaml


class CanvasParameters:
    def __init__(self, config_path: str | Path):
        """
        Load a configuration file and create a simple object container

        :param config_path: str or Path to a configuration file (yaml)
        """
        # Load parameters
        parameters_dict: dict = load_yaml(config_path)

        # Resize logo
        self.logo_side = parameters_dict['logo_side_resize']

        # Specify grid dimensions
        self.rows: int = parameters_dict["rows"]
        self.cols: int = parameters_dict["cols"]

        # Specify distance between logos
        self.spacing: int = parameters_dict["spacing"]
        self.skew: float = parameters_dict["interval_skew_fraction"]

        # Specify transformation factors
        self.transform_prob: float = parameters_dict["transform_prob"]
        self.translation_factor: float = parameters_dict["translation_factor"]
        self.rotation_factor: float = parameters_dict["rotation_factor"]
        self.scale_factor: float = parameters_dict["scale_factor"]

        # Gaussian noise parameters (basically mean and std)
        self.min_noise: float = parameters_dict["min_noise"]
        self.max_noise: float = parameters_dict["max_noise"]

        self.logo_contrast_min: float = parameters_dict["logo_contrast_min"]
        self.logo_contrast_max: float = parameters_dict["logo_contrast_max"]
        self.contrast_min: float = parameters_dict["contrast_min"]
        self.contrast_max: float = parameters_dict["contrast_max"]
        self.band_percentage: float = parameters_dict["band_percentage"]

        # Background
        self.bg_color = parameters_dict["bg_color"]
        self.bg_texture = ""
        self.use_texture = parameters_dict["use_texture"]
