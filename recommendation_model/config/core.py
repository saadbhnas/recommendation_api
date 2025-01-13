# -*- coding: utf-8 -*-

import os
from pathlib import Path 
import sys 

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import recommendation_model
from typing import  List, Optional 
import pathlib
from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load

Package_root = pathlib.Path(recommendation_model.__file__).absolute().parent
dataset_folder = Package_root/"dataset"
root = Package_root.parent
CONFIG_FILE_PATH = Package_root/'config.yml'
trained_model_dir = Package_root/'training_dir'


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    #test_data_file: str
    pipeline_save_file: str
    similiarity_score:str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    #target: str
    vars_extract_name: List[str]
    movies_metadata_features: List[str]
    variables_drop:List[str]
    variables_to_replace:List[str]
    overview_replace:List[str]
    popularity_replace:List[str]
    budget_replace:List[str]
    changing_vars_type:List[str]
    creating_content_column:List[str]
    credits_features:List[str]
    keywords_features:List[str]
    movies_metadata_json_variables:List[str]
    credits_json_variables:List[str]
    keywords_json_varibale:List[str]
    credits_vars_extract_names:List[str]
    test_size: float
    random_state: int
    strings_to_replace_overview:List[str]
    strings_to_replace_budget:List[str]
    adjust_budget_revenue_scale:int
    strings_to_replace_popularity:str
    
    """
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: Sequence[str]
    binarize_vars: Sequence[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]
"""

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),   # here **  unpacking the file into key value pair and pass them to AppConfig
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()


