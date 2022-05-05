#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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

import io
import time
import torch
from typing import Optional, Union, Text
from pathlib import Path
from os.path import basename
import numpy as np

from pyannote.database import FileFinder
from pretrained import Pretrained
from config import load_config
from features.with_librosa import LibrosaMFCC
import copy
from core import SlidingWindow
from core import SlidingWindowFeature
import scipy.signal
from core import Segment, Timeline
from core.utils.generators import pairwise

class Peak(object):
    """Peak detection

    Parameters
    ----------
    alpha : float, optional
        Adaptative threshold coefficient. Defaults to 0.5
    scale : {'absolute', 'relative', 'percentile'}
        Set to 'relative' to make onset/offset relative to min/max.
        Set to 'percentile' to make them relative 1% and 99% percentiles.
        Defaults to 'absolute'.
    min_duration : float, optional
        Defaults to 1 second.
    log_scale : bool, optional
        Set to True to indicate that binarized scores are log scaled.
        Defaults to False.

    """

    def __init__(self, alpha=0.5, min_duration=1.0):
        super(Peak, self).__init__()
        self.alpha = alpha
        self.min_duration = min_duration


    def apply(self, predictions, dimension=0):
        """Peak detection

        Parameter
        ---------
        predictions : SlidingWindowFeature
            Predictions returned by segmentation approaches.

        Returns
        -------
        segmentation : Timeline
            Partition.
        """

        if len(predictions.data.shape) == 1:
            y = predictions.data
        elif predictions.data.shape[1] == 1:
            y = predictions.data[:, 0]
        else:
            y = predictions.data[:, dimension]

        sw = predictions.sliding_window

        precision = sw.step
        order = max(1, int(np.rint(self.min_duration / precision)))
        indices = scipy.signal.argrelmax(y, order=order)[0]

        threshold = self.alpha 

        peak_time = np.array([sw[i].middle for i in indices if y[i] > threshold])

        n_windows = len(y)
        start_time = sw[0].start
        end_time = sw[n_windows].end

        boundaries = np.hstack([[start_time], peak_time, [end_time]])
        segmentation = Timeline()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation.add(segment)

        return segmentation


class Application():

    @classmethod
    def from_train_dir(cls, train_dir: Path, training: bool = False):

        app = cls(train_dir.parents[1], training=training)
        app.train_dir_ = train_dir
        return app

    def __init__(
        self,
    ):
        """

        Parameters
        ----------
        experiment_dir : Path
        training : boolean, optional
            When False, data augmentation is disabled.
        pretrained_config_yml : Path, optional
        """

        # load configuration
        config_yml = 'config.yml'
        config_default_module = getattr(
            self, "config_default_module", "pyannote.audio.labeling.tasks"
        )

        config = load_config(
            Path(config_yml),
            training=False,
            config_default_module=config_default_module,
            pretrained_config_yml=None,
        )
        self.config = config
        self.non_speech = config['non_speech']
        for key, value in config.items():
            setattr(self, f"{key}_", value)


    def run(
        self,
        wav,
        sr,
        chronological: bool = False,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        down_rate: int = 1,
        step: float = 0.25,
        best_threshold: float = None,
        **kwargs,
    ):
        mfcc = self.feature_extraction_(wav, sr)
        # compute (and store) SCD scores
        pretrained = Pretrained(
            duration=4.0,
            step=step,
            batch_size=batch_size,
            device=device,
            down_rate=down_rate,
        )
        scores_ = pretrained(mfcc)
        scores = copy.deepcopy(scores_)
        if down_rate != 1:
            scores.sliding_window._SlidingWindow__duration = scores.sliding_window._SlidingWindow__duration + (down_rate - 1) * scores.sliding_window._SlidingWindow__step
            scores.sliding_window._SlidingWindow__step = scores.sliding_window._SlidingWindow__step * down_rate

        peak = Peak(alpha=best_threshold)
        change_prob = SlidingWindowFeature(scores.data[:, -1], scores.sliding_window)
        change = peak.apply(change_prob)
        print('done!')
        return change
