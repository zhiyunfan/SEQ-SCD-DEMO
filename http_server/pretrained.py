# The MIT License (MIT)
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr

from utils import RawAudio
from typing import Optional
from typing import Union
from typing import Text

import torch
import numpy as np

from core.segment import SlidingWindow
from core.feature import SlidingWindowFeature
from config import load_config
from config import load_specs

RESOLUTION_FRAME = "frame"
RESOLUTION_CHUNK = "chunk"


class Pretrained:
    """
    Parameters
    ----------
    validate_dir : Path
        Path to a validation directory.
    epoch : int, optional
        If provided, force loading this epoch.
        Defaults to reading epoch in validate_dir/params.yml.
    augmentation : Augmentation, optional
    duration : float, optional
        Use audio chunks with that duration. Defaults to the fixed duration
        used during training, when available.
    step : float, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.25.
    device : optional
    return_intermediate : optional
    """

    # TODO: add progress bar (at least for demo purposes)

    def __init__(
        self,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
        device: Optional[Union[Text, torch.device]] = None,
        down_rate=8,
    ):
        self.down_rate = down_rate
        self.duration_ = duration
        self.step = step

        config_yml = "config.yml"
        config = load_config(config_yml, training=False)
        # use feature extraction from config.yml configuration file
        self.feature_extraction_ = config["feature_extraction"]

        self.sample_rate = 16000

        self.raw_audio_ = RawAudio(
            sample_rate=self.sample_rate, mono=True, augmentation=None
        )

        specs_yml = "specs.yml"
        specifications = load_specs(specs_yml)

        self.preprocessors_ = config["preprocessors"]

        self.weights_pt_ = "./models/model.pt"

        specifications['training'] = False
        model = config["get_model_from_specs"](specifications)
        model.load_state_dict(
            torch.load(self.weights_pt_, map_location=lambda storage, loc: storage)
        )

        # defaults to using GPU when available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # send model to device
        self.model_ = model.eval().to(self.device)

        # initialize chunks duration with that used during training
        #self.duration = getattr(config["task"], "duration", None)

        self.min_duration = getattr(config["task"], "min_duration", None)

        if step is None:
            step = 0.25
        self.step = step
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )
        self.batch_size = batch_size


    @property
    def duration(self):
        return self.duration_

    @duration.setter
    def duration(self, duration: float):
        self.duration_ = duration
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )

    @property
    def step(self):
        return getattr(self, "step_", 0.25)

    @step.setter
    def step(self, step: float):
        self.step_ = step
        self.chunks_ = SlidingWindow(
            duration=self.duration, step=self.step * self.duration
        )


    def get_resolution(self) -> SlidingWindow:

        resolution = self.model_.resolution
        # model returns one vector per input frame
        if resolution == RESOLUTION_FRAME:
            resolution = self.feature_extraction_.sliding_window

        # model returns one vector per input window
        if resolution == RESOLUTION_CHUNK:
            resolution = self.chunks_
        return resolution

    def __call__(self, features) -> SlidingWindowFeature:
        """Extract features from file

        Parameters
        ----------
        current_file : dict
            `pyannote.database` files.

        Returns
        -------
        features : `pyannote.core.SlidingWindowFeature`
            Extracted features
        """
        features = self.get_features(features, self.sample_rate)

        # wrap features in a `SlidingWindowFeature` instance
        return SlidingWindowFeature(features, self.get_resolution())

    def get_features(self, y, sample_rate) -> np.ndarray:
        # import pdb
        # pdb.set_trace() 
        # features = SlidingWindowFeature(
        #     y,
        #     self.feature_extraction_.sliding_window,
        # )
        #if 'y' in current_file.keys():
        return self.model_.slide(
            y,
            self.chunks_,
            batch_size=self.batch_size,
            device=self.device,
            down_rate=self.down_rate,
        ).data
