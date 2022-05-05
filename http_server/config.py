#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import warnings
from functools import partial

from pathlib import Path
import shutil
from typing import Text
from typing import Dict
import yaml

import collections
from pyannote.database import FileFinder
from task import Task
from typing import Optional
from importlib import import_module


def get_class_by_name(class_name: str,
                      default_module_name: Optional[str] = None) -> type:
    """Load class by its name

    Parameters
    ----------
    class_name : `str`
    default_module_name : `str`, optional
        When provided and `class_name` does not contain the absolute path

    Returns
    -------
    Klass : `type`
        Class.

    Example
    -------
    >>> ClopiNet = get_class_by_name(
    ...     'pyannote.audio.embedding.models.ClopiNet')
    >>> ClopiNet = get_class_by_name(
    ...     'ClopiNet', default_module_name='pyannote.audio.embedding.models')
    """
    tokens = class_name.split('.')

    if len(tokens) == 1:
        if default_module_name is None:
            msg = (
                f'Could not infer module name from class name "{class_name}".'
                f'Please provide default module name.')
            raise ValueError(msg)
        module_name = default_module_name
    else:
        module_name = '.'.join(tokens[:-1])
        class_name = tokens[-1]

    return getattr(import_module(module_name), class_name)


def merge_cfg(pretrained_cfg, cfg):
    for k, v in cfg.items():

        # case where the user purposedly set a section value to "null"
        # this might happen when fine-tuning a pretrained model
        if v is None:
            _ = pretrained_cfg.pop(k, None)

        # if v is a dictionary, go deeper and merge recursively
        elif isinstance(v, collections.abc.Mapping):
            pretrained_cfg[k] = merge_cfg(pretrained_cfg.get(k, {}), v)

        # in any other case, override pretrained_cfg[k] by cfg[k]
        else:
            pretrained_cfg[k] = v

    return pretrained_cfg


def load_config(
    config_yml: Path,
    training: bool = False,
    config_default_module: Text = None,
    pretrained_config_yml: Path = None,
) -> Dict:
    """

    Returns
    -------
    config : Dict
        ['preprocessors']
        ['learning_rate']
        ['scheduler']
        ['get_optimizer']
        ['callbacks']
        ['feature_extraction']
        ['task']
        ['get_model_from_specs']
        ['model_resolution']
        ['model_alignment']
    """

    # load pretrained model configuration
    pretrained_cfg = dict()
    if pretrained_config_yml is not None:
        with open(pretrained_config_yml, "r") as fp:
            pretrained_cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    # load configuration or complain it's missing
    cfg = dict()

    with open(config_yml, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    # backup user-provided config because it will be updated
    if pretrained_config_yml is not None:
        shutil.copy(config_yml, config_yml.parent / "backup+config.yml")


    # override pretrained model config with user-provided config
    cfg = merge_cfg(pretrained_cfg, cfg)

    # save (updated) config to disk
    if pretrained_config_yml is not None:
        with open(config_yml, "w") as fp:
            yaml.dump(cfg, fp, default_flow_style=False)

    # preprocessors
    preprocessors = dict()

    for key, preprocessor in cfg.get("preprocessors", {}).items():
        # preprocessors:
        #    key:
        #       name: package.module.ClassName
        #       params:
        #          param1: value1
        #          param2: value2
        if isinstance(preprocessor, dict):
            Klass = get_class_by_name(preprocessor["name"])
            preprocessors[key] = Klass(**preprocessor.get("params", {}))
            continue

        try:
            # preprocessors:
            #    key: /path/to/database.yml
            preprocessors[key] = FileFinder(database_yml=preprocessor)

        except FileNotFoundError as e:
            # preprocessors:
            #    key: /path/to/{uri}.wav
            preprocessors[key] = preprocessor

    cfg["preprocessors"] = preprocessors

    OPTIMIZER_DEFAULT = {
        "name": "Adam",
        "params": {
            "lr": 0.001,
            "betas": (0.9, 0.999),
            "weight_decay": 0,
            "eps": 1e-08,
        },
    }
    optimizer_cfg = cfg.get("optimizer", OPTIMIZER_DEFAULT)
    try:
        Optimizer = get_class_by_name(
            optimizer_cfg["name"], default_module_name="torch.optim"
        )
        optimizer_params = optimizer_cfg.get("params", {})
        cfg["get_optimizer"] = partial(Optimizer, **optimizer_params)

    # do not raise an error here as it is possible that the optimizer is
    # not really needed (e.g. in pipeline training)
    except ModuleNotFoundError as e:
        warnings.warn(e.args[0])

    # data augmentation should only be active when training a model
    if training and "data_augmentation" in cfg:
        DataAugmentation = get_class_by_name(
            cfg["data_augmentation"]["name"],
            default_module_name="pyannote.audio.augmentation",
        )
        augmentation = DataAugmentation(**cfg["data_augmentation"].get("params", {}))
    else:
        augmentation = None


    # feature extraction
    FEATURE_DEFAULT = {"name": "RawAudio", "params": {"sample_rate": 16000}}
    feature_cfg = cfg.get("feature_extraction", FEATURE_DEFAULT)
    FeatureExtraction = get_class_by_name(
        feature_cfg["name"], default_module_name="features"
    )
    feature_params = feature_cfg.get("params", {})
    cfg["feature_extraction"] = FeatureExtraction(
        **feature_params, augmentation=augmentation
    )

    # # task
    # if config_default_module is None:
    #     config_default_module = "tasks"
    # try:
    #     TaskClass = get_class_by_name(
    #         cfg["task"]["name"], default_module_name=config_default_module
    #     )
    # except AttributeError:
    #     TaskClass = get_class_by_name(
    #         cfg["task"]["name"],
    #         default_module_name="pyannote.audio.embedding.approaches",
    #     )
    cfg["collar"] = cfg["task"]["params"]["collar"]
    cfg["non_speech"] = cfg["task"]["params"]["non_speech"]
    #cfg["task"] = TaskClass(**cfg["task"].get("params", {}))
 

    # architecture
    Architecture = get_class_by_name(
        cfg["architecture"]["name"], default_module_name="models"
    )
    params = cfg["architecture"].get("params", {})

    cfg["get_model_from_specs"] = partial(Architecture, **params)
    # task = cfg["task"].task
    # cfg["model_resolution"] = Architecture.get_resolution(task, **params)
    # cfg["model_alignment"] = Architecture.get_alignment(task, **params)

    return cfg


def load_specs(specs_yml: Path) -> Dict:
    """

    Returns
    -------
    specs : Dict
        ['task']
        [and others]
    """

    with open(specs_yml, "r") as fp:
        specifications = yaml.load(fp, Loader=yaml.SafeLoader)
    specifications["task"] = Task.from_str(specifications["task"])
    return specifications


def load_params(params_yml: Path) -> Dict:

    with open(params_yml, "r") as fp:
        params = yaml.load(fp, Loader=yaml.SafeLoader)

    return params
