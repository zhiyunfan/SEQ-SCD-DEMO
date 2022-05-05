#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019 CNRS

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
# Juan Manuel Coria

from typing import Optional
from typing import Text
import torch
import torch.nn as nn
from tdnn import TDNN
from linear import Linear

from task import Task
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple
import logging
import math
import numpy as np
import time
from typing import Union
from core import SlidingWindow
from core import SlidingWindowFeature
from torch.nn import Module
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import pescador
import copy


RESOLUTION_FRAME = "frame"
RESOLUTION_CHUNK = "chunk"
Resolution = Union[SlidingWindow, Literal[RESOLUTION_FRAME, RESOLUTION_CHUNK]]


ALIGNMENT_CENTER = "center"
ALIGNMENT_STRICT = "strict"
ALIGNMENT_LOOSE = "loose"
Alignment = Literal[ALIGNMENT_CENTER, ALIGNMENT_STRICT, ALIGNMENT_LOOSE]

class Model(Module):
    """Model

    A `Model` is nothing but a `torch.nn.Module` instance with a bunch of
    additional methods and properties specific to `pyannote.audio`.

    It is expected to be instantiated with a unique `specifications` positional
    argument describing the task addressed by the model, and a user-defined
    number of keyword arguments describing the model architecture.

    Parameters
    ----------
    specifications : `dict`
        Task specifications.
    **architecture_params : `dict`
        Architecture hyper-parameters.
    """

    def __init__(self, specifications: dict, **architecture_params):
        super().__init__()
        self.specifications = specifications
        training = specifications['training']
        architecture_params['training'] = training
        self.resolution_ = self.get_resolution(self.task, **architecture_params)
        self.alignment_ = self.get_alignment(self.task, **architecture_params)
        self.init(**architecture_params)


        self.n_batches_ = 0
        self.loss_moving_avg_ = dict()
        self.beta_ = 0.98


    def init(self, **architecture_params):
        """Initialize model architecture

        This method is called by Model.__init__ after attributes
        'specifications', 'resolution_', and 'alignment_' have been set.

        Parameters
        ----------
        **architecture_params : `dict`
            Architecture hyper-parameters

        """
        msg = 'Method "init" must be overriden.'
        raise NotImplementedError(msg)

    @property
    def probes(self):
        """Get list of probes"""
        return list(getattr(self, "_probes", []))

    @probes.setter
    def probes(self, names: List[Text]):
        """Set list of probes

        Parameters
        ----------
        names : list of string
            Names of modules to probe.
        """

        for handle in getattr(self, "handles_", []):
            handle.remove()

        self._probes = []

        if not names:
            return

        handles = []

        def _init(module, input):
            self.probed_ = dict()

        handles.append(self.register_forward_pre_hook(_init))

        def _append(name, module, input, output):
            self.probed_[name] = output

        for name, module in self.named_modules():
            if name in names:
                handles.append(module.register_forward_hook(partial(_append, name)))
                self._probes.append(name)

        def _return(module, input, output):
            return output, self.probed_

        handles.append(self.register_forward_hook(_return))

        self.handles_ = handles

    @probes.deleter
    def probes(self):
        """Remove all probes"""
        for handle in getattr(self, "handles_", []):
            handle.remove()
        self._probes = []

    @property
    def parts(self):
        """Names of (freezable / probable) modules"""
        return [n for n, _ in self.named_modules()]

    def freeze(self, names: List[Text]):
        """Freeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to freeze.
        """
        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = False

    def unfreeze(self, names: List[Text]):
        """Unfreeze parts of the model

        Parameters
        ----------
        names : list of string
            Names of modules to unfreeze.
        """

        for name, module in self.named_modules():
            if name in names:
                for parameter in module.parameters(recurse=True):
                    parameter.requires_grad = True

    def forward(
        self, sequences: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[Text, torch.Tensor]]]:
        """TODO

        Parameters
        ----------
        sequences : (batch_size, n_samples, n_features) `torch.Tensor`
        **kwargs : `dict`

        Returns
        -------
        output : (batch_size, ...) `torch.Tensor`
        probes : dict, optional
        """

        # TODO
        msg = "..."
        raise NotImplementedError(msg)

    @property
    def task(self) -> Task:
        """Type of task addressed by the model

        Shortcut for self.specifications['task']
        """
        return self.specifications["task"]

    def get_resolution(self, task: Task, **architecture_params) -> Resolution:
        """Get frame resolution

        This method is called by `BatchGenerator` instances to determine how
        target tensors should be built.

        Depending on the task and the architecture, the output of a model will
        have different resolution. The default behavior is to return
        - `RESOLUTION_CHUNK` if the model returns just one output for the whole
          input sequence
        - `RESOLUTION_FRAME` if the model returns one output for each frame of
          the input sequence

        In case neither of these options is valid, this method needs to be
        overriden to return a custom `SlidingWindow` instance.

        Parameters
        ----------
        task : Task
        **architecture_params
            Parameters used for instantiating the model architecture.

        Returns
        -------
        resolution : `Resolution`
            - `RESOLUTION_CHUNK` if the model returns one single output for the
              whole input sequence;
            - `RESOLUTION_FRAME` if the model returns one output for each frame
               of the input sequence.
        """

        if task.returns_sequence:
            return RESOLUTION_FRAME

        elif task.returns_vector:
            return RESOLUTION_CHUNK

        else:
            # this should never happened
            msg = f"{task} tasks are not supported."
            raise NotImplementedError(msg)

    @property
    def resolution(self) -> Resolution:
        return self.resolution_

    def get_alignment(self, task: Task, **architecture_params) -> Alignment:
        """Get frame alignment

        This method is called by `BatchGenerator` instances to dermine how
        target tensors should be aligned with the output of the model.

        Default behavior is to return 'center'. In most cases, you should not
        need to worry about this but if you do, this method can be overriden to
        return 'strict' or 'loose'.

        Parameters
        ----------
        task : Task
        architecture_params : dict
            Architecture hyper-parameters.

        Returns
        -------
        alignment : `Alignment`
            Target alignment. Must be one of 'center', 'strict', or 'loose'.
            Always returns 'center'.
        """

        return ALIGNMENT_CENTER

    @property
    def alignment(self) -> Alignment:
        return self.alignment_

    @property
    def n_features(self) -> int:
        """Number of input features

        Shortcut for self.specifications['X']['dimension']

        Returns
        -------
        n_features : `int`
            Number of input features
        """
        return self.specifications["X"]["dimension"]

    @property
    def dimension(self) -> int:
        """Output dimension

        This method needs to be overriden for representation learning tasks,
        because output dimension cannot be inferred from the task
        specifications.

        Returns
        -------
        dimension : `int`
            Dimension of model output.

        Raises
        ------
        AttributeError
            If the model addresses a classification or regression task.
        """

        if self.task.is_representation_learning:
            msg = (
                f"Class {self.__class__.__name__} needs to define "
                f"'dimension' property."
            )
            raise NotImplementedError(msg)

        msg = f"{self.task} tasks do not define attribute 'dimension'."
        raise AttributeError(msg)

    @property
    def classes(self) -> List[str]:
        """Names of classes

        Shortcut for self.specifications['y']['classes']

        Returns
        -------
        classes : `list` of `str`
            List of names of classes.


        Raises
        ------
        AttributeError
            If the model does not address a classification task.
        """

        if not self.task.is_representation_learning:
            return self.specifications["y"]["classes"]

        msg = f"{self.task} tasks do not define attribute 'classes'."
        raise AttributeError(msg)


    def crop_y(self, y, segment):
        """Extract y for specified segment

        Parameters
        ----------
        y : `pyannote.core.SlidingWindowFeature`
            Output of `initialize_y` above.
        segment : `pyannote.core.Segment`
            Segment for which to obtain y.

        Returns
        -------
        cropped_y : (n_samples, dim) `np.ndarray`
            y for specified `segment`
        """

        return y.crop(segment, mode="center", fixed=4)


    def slide(
        self,
        features: SlidingWindowFeature,
        sliding_window: SlidingWindow,
        batch_size: int = 32,
        device: torch.device = None,
        down_rate=8,
    ) -> SlidingWindowFeature:
        """Slide and apply model on features

        Parameters
        ----------
        features : SlidingWindowFeature
            Input features.
        sliding_window : SlidingWindow
            Sliding window used to apply the model.
        batch_size : int
            Batch size. Defaults to 32. Use large batch for faster inference.
        device : torch.device
            Device used for inference.
        skip_average : bool, optional
            For sequence labeling tasks (i.e. when model outputs a sequence of
            scores), each time step may be scored by several consecutive
            locations of the sliding window. Default behavior is to average
            those multiple scores. Set `skip_average` to False to return raw
            scores without averaging them.
        postprocess : callable, optional
            Function applied to the predictions of the model, for each batch
            separately. Expects a (batch_size, n_samples, n_features) np.ndarray
            as input, and returns a (batch_size, n_samples, any) np.ndarray.
        return_intermediate :
            Experimental. Not documented yet.
        progress_hook : callable
            Experimental. Not documented yet.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        try:
            dimension = self.dimension
        except AttributeError:
            dimension = len(self.classes)

        resolution = self.resolution

        resolution = features.sliding_window

        support = features.extent
        if support.duration < sliding_window.duration:
            chunks = [support]
            fixed = support.duration
        else:
            chunks = list(sliding_window(support, align_last=True))
            fixed = sliding_window.duration

        batches = pescador.maps.buffer_stream(
            iter(
                 {"X": features.crop(window, mode="center", fixed=fixed)}
                for window in chunks
            ),
             batch_size,
            partial=True,
        )
        fX = []
        for batch in batches:
            tX = torch.tensor(batch["X"], dtype=torch.float32, device=device)

            # FIXME: fix support for return_intermediate
            with torch.no_grad():
                cif_output, output, not_padding_after_cif, sum_a, fired_flag, loss_cfg = self(tX, mask=None)

            tfX = torch.cat([fired_flag[:,:,-1].unsqueeze(-1), fired_flag[:,:,-1].unsqueeze(-1)], -1)


            tfX_npy = tfX.detach().to("cpu").numpy()

            fX.append(tfX_npy)


        fX = np.vstack(fX)

        # get total number of frames (based on last window end time)
        resolution_ = copy.copy(resolution)
        if down_rate != 1:
            resolution_._SlidingWindow__duration = resolution._SlidingWindow__duration + (down_rate - 1) * resolution._SlidingWindow__step
            resolution_._SlidingWindow__step = resolution._SlidingWindow__step * down_rate

        n_frames = resolution_.samples(chunks[-1].end, mode="center")
        # data[i] is the sum of all predictions for frame #i
        data = np.zeros((n_frames, dimension), dtype=np.float32)

        # k[i] is the number of chunks that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        for chunk, fX_ in zip(chunks, fX):
            # indices of frames overlapped by chunk
            indices = resolution_.crop(chunk, mode=self.alignment, fixed=fixed)

            # accumulate the outputs
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            # TODO - use smarter weights (e.g. Hamming window)
            k[indices] += 1
        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        def process(data):
            for t in range(1, data.shape[0]-1):
                if data[t+1][-1] != 0 and data[t][-1] != 0:
                    value = data[t]
                    u = t
                    while (u < data.shape[0] -1 and data[u+1][-1]!=0):
                        u += 1   
                    peak = (t + u) // 2
                    tmp1 = np.sum(data[t:u+1, 0])
                    tmp2 = np.sum(data[t:u+1, 1])
                    data[t:u+1,0] = 1
                    data[t:u+1,1] = 0
                    data[peak,0] = tmp1
                    data[peak,1] = tmp2
            return None

        process(data)
        return SlidingWindowFeature(data, resolution_)

class Encoder(nn.Module):
    """Recurrent layers

    Parameters
    ----------
    n_features : `int`
        Input feature shape.
    unit : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    hidden_size : `int`, optional
        Number of features in the hidden state h. Defaults to 16.
    num_layers : `int`, optional
        Number of recurrent layers. Defaults to 1.
    bias : `boolean`, optional
        If False, then the layer does not use bias weights. Defaults to True.
    dropout : `float`, optional
        If non-zero, introduces a Dropout layer on the outputs of each layer
        except the last layer, with dropout probability equal to dropout.
        Defaults to 0.
    bidirectional : `boolean`, optional
        If True, becomes a bidirectional RNN. Defaults to False.
    concatenate : `boolean`, optional
        Concatenate output of each layer instead of using only the last one
        (which is the default behavior).
    pool : {'sum', 'max', 'last', 'x-vector'}, optional
        Temporal pooling strategy. Defaults to no pooling.
    """

    def __init__(
        self,
        n_features,
        down_rate,
        unit="LSTM",
        hidden_size=256,
        num_layers=1,
        bias=True,
        dropout=0,
        bidirectional=False,
    ):
        super().__init__()

        self.n_features = n_features
        self.unit = unit
        Klass = getattr(nn, self.unit)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.down_rate = down_rate
        
        self.rnn_ = nn.ModuleList([])
        self.tdnn_ = nn.ModuleList([])
        
        if self.down_rate == 1:
            stride = [1,1,1,1]
        elif self.down_rate == 2:
            stride = [1,1,1,2]
        elif self.down_rate == 4:
            stride = [1,1,2,2]
        elif self.down_rate == 8:
            stride = [1,2,2,2]
        elif self.down_rate == 16:
            stride = [2,2,2,2]
        elif self.down_rate == 32:
            stride = [2,2,4,2]
        else:
            raise ValueError(f'wrong down rate')

        for i in range(len(stride)):
            if i == 0:
                input_dim = n_features
            else:
                input_dim = 2 * self.hidden_size
            
            out_dim = 2 * self.hidden_size

            tdnn = TDNN(
                context=[-2, 2],
                input_channels=input_dim,
                output_channels=out_dim,
                full_context=True,
                stride=stride[i],
                padding=2,
            )

            self.tdnn_.append(tdnn)
            self.n_features = out_dim
  
        for i in range(self.num_layers):
            if i + 1 == self.num_layers:
                dropout = 0
            else:
                dropout = self.dropout

            rnn = Klass(
                2 * self.hidden_size,
                self.hidden_size,
                num_layers=1,
                bias=self.bias,
                batch_first=True,
                dropout=dropout,
                bidirectional=self.bidirectional,
            )
          
            self.rnn_.append(rnn)
            
    def forward(self, features, return_intermediate=False):
        """Apply recurrent layer (and optional temporal pooling)

        Parameters
        ----------
        features : `torch.Tensor`
            Features shaped as (batch_size, n_frames, n_features)
        return_intermediate : `boolean`, optional
            Return intermediate RNN hidden state.

        Returns
        -------
        output : `torch.Tensor`
            TODO. Shape depends on parameters...
        intermediate : `torch.Tensor`
            (num_layers, batch_size, hidden_size * num_directions)
        """
        hidden = None
        output = features

        for j in range(3):
            output = self.tdnn_[j](output)

        output, hidden = self.rnn_[0](output, hidden)
        output = self.tdnn_[3](output)
        output, hidden = self.rnn_[1](output)

        return output

    def dimension():
        doc = "Output features dimension."

        def fget(self):
            if self.num_layers < 1:
                dimension = self.n_features
            else:
                dimension = self.hidden_size

            if self.bidirectional:
                dimension *= 2

            return dimension

        return locals()

    dimension = property(**dimension())

    def intermediate_dimension(self, layer):
        if self.num_layers < 1:
            dimension = self.n_features
        else:
            dimension = self.hidden_size

        if self.bidirectional:
            dimension *= 2

        return dimension

class CIF(nn.Module):
    def __init__(
        self,
        cif_weight_threshold=1,
        encoder_out_dim=256,
        max_history=10,
        weight_active='crelu',
        relu_threshold=1.0,
        using_scaling=True,
        nonlinear_act=None,
        using_kaiming_init=False,
        using_bias_constant_init=False,
        normalize_scalar=1.0,
        ):
        super().__init__()
        self.cif_weight_threshold = cif_weight_threshold
        self.max_history = max_history
        self.weight_active = weight_active
        self.relu_threshold = relu_threshold
        self.using_scaling = using_scaling
        self.nonlinear_act = nonlinear_act
        self.normalize_scalar = normalize_scalar

        n = 2

        self.weight_dense1 = nn.Linear(n * encoder_out_dim, n * encoder_out_dim)
        self.weight_dense2 = nn.Linear(n * encoder_out_dim, 1)

        if using_bias_constant_init == True:
            nn.init.constant_(self.weight_dense1.bias, 0.05)
            nn.init.constant_(self.weight_dense2.bias, 0.05)

        if using_kaiming_init == True:
            nn.init.kaiming_normal_(self.weight_dense1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_normal_(self.weight_dense2.weight, mode='fan_in', nonlinearity='relu')

        if self.nonlinear_act == 'relu':
            self.diff_act = torch.nn.ReLU()
        elif self.nonlinear_act == 'tanh':
            self.diff_act = torch.nn.Tanh()
        else:
            pass

        if weight_active == 'crelu':
            self.crelu = torch.nn.ReLU()

        if weight_active == 'leakyrelu' or weight_active == 'leakyrelu_nc' :
            self.leakyrelu = torch.nn.LeakyReLU(0.01)


    def forward(self, encoder_outputs, mask, is_training):
        """The CIF part in between the encoder and decoder,
           firing the representation of one acoustic unit
           when the information is accumulated over the given threshold.

        Args:
          encoder_outputs: a Tensor of shape [batch_size, down_sampled_length, hidden_size].
          not_padding: shows the effective and non-effective, with shape [batch_size, down_sampled_length].
          is_training: a boolean.
          targets: the batch of targets used for score scaling, with shape [batch_size, target_length]


        Returns:
          cif_outputs: the output of the CIF part, with shape [batch_size, cif_output_length, hidden_size]
          not_padding_after_cif: shows the effective and non-effective.
          sum_a: the sum of predicted score of each high-level frame, used for loss calculation.
          integrated_logits_on_encoder: the integrated logits by using the same weight estimated by CIF,
                                        with shape [batch_size, cif_output_length, vocab_size]
        """
        encoder_outputs_ori = encoder_outputs
        first_padding_pos = (torch.sum(torch.abs(encoder_outputs), -1) != 0.0).int().sum(1)

        threshold = self.cif_weight_threshold
        batch_size = encoder_outputs.size(0)
        encoder_output_length = encoder_outputs.size(1)
        hidden_size = encoder_outputs.size(2)

        accumulated_weights = torch.zeros([batch_size, 0]).cuda()
        accumulated_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_states = torch.zeros([batch_size, 0, hidden_size]).cuda()
        fired_flag = torch.zeros([batch_size, 0, 1]).cuda()
        
        history_chunk_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()
        history_std_keep = torch.zeros(batch_size, 0 , hidden_size).cuda()

        for t in range(encoder_output_length):
            if t == 0:
                history_chunk = encoder_outputs[:, 0, :]
            else:
                start = max(0, t-self.max_history)
                history_chunk = torch.mean(encoder_outputs[:, start:t, :], 1)
            
            # fuse future information
            history_chunk_keep = torch.cat([history_chunk_keep, history_chunk.unsqueeze(1)], 1)

        #计算差异量
        ht = encoder_outputs

        his_diff = ht - history_chunk_keep

        info = his_diff
        info = torch.cat([encoder_outputs, info], -1)
 
        weight = self.weight_dense1(info)

        if self.nonlinear_act != None:
            weight = self.diff_act(weight)
        
        weight = self.weight_dense2(weight)

        if self.weight_active == 'hardsigmoid':
            weight = torch.nn.functional.hardsigmoid(weight)
        elif self.weight_active == 'sigmoid':
            weight = torch.sigmoid(weight)
        elif self.weight_active == 'crelu':
            weight = self.crelu(weight)
            weight = torch.clip(weight, max=self.relu_threshold)
        else:
            pass
        weight = weight.squeeze(-1)

        # keep original weight
        ori_weight = weight

        # scaling
        if is_training and self.using_scaling and mask != None:
            sum_a = weight.sum(1)
            N = mask[:,:,0].sum(-1) - 1
            scale = N/torch.clip(sum_a, min=1e-8)
            scale = torch.where(sum_a == 0,
                                torch.zeros_like(sum_a),
                                scale)
            #print(N, sum_a, scale)
            weight = scale.unsqueeze(-1).repeat(1, encoder_output_length) * weight
        else:
            sum_a = weight.sum(1)


        for t in range(encoder_output_length):
            if t == 0:
                prev_accumulated_weight = torch.zeros([batch_size]).cuda()
                prev_accumulated_state = encoder_outputs[:, 0, :]
            else:
                prev_accumulated_weight = accumulated_weights[:, t-1]
                prev_accumulated_state = accumulated_states[:, t-1, :]
 
            cur_weight = weight[:,t]

            cur_weight = torch.where(torch.full([batch_size], t).cuda() > first_padding_pos,
                                       torch.zeros([batch_size]).cuda(),
                                       cur_weight)

            cur_is_fired = prev_accumulated_weight + cur_weight > threshold
            cur_is_fired = cur_is_fired.int()
            remained_weight = threshold - prev_accumulated_weight
            cur_accumulated_weight = torch.where(cur_is_fired == 1,
                                            cur_weight - remained_weight,
                                            cur_weight + prev_accumulated_weight)

            cur_accumulated_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                                encoder_outputs[:, t, :],
                                                prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :])
            cur_fired_state = torch.where(cur_is_fired.unsqueeze(-1).repeat(1, hidden_size)==1,
                                        prev_accumulated_state + (1-cur_weight.unsqueeze(-1))*encoder_outputs[:, t, :],
                                        torch.zeros([batch_size, hidden_size]).cuda()) 
             
            # handling the speech tail by rounding up and down
            cur_fired_state = torch.where(torch.full([batch_size, 1], t).cuda() > first_padding_pos.unsqueeze(-1),
                                            torch.zeros([batch_size, hidden_size]).cuda(),
                                            cur_fired_state) 

            fired_flag = torch.cat([fired_flag, cur_is_fired.unsqueeze(-1).unsqueeze(-1)], 1)
            accumulated_weights = torch.cat([accumulated_weights, cur_accumulated_weight.unsqueeze(-1)], 1)
            accumulated_states = torch.cat([accumulated_states, cur_accumulated_state.unsqueeze(1)], 1)                     
            fired_states = torch.cat([fired_states, cur_fired_state.unsqueeze(1)], 1)

        tail_fired_states = cur_accumulated_state
        fired_states = torch.cat([fired_states, tail_fired_states.unsqueeze(1)], 1)

        fired_marks = (torch.sum(torch.abs(fired_states), -1) != 0.0).int()
        fired_utt_length = fired_marks.sum(-1)
        fired_max_length = torch.max(fired_utt_length).int()

        cif_outputs = torch.zeros([0, fired_max_length, hidden_size]).cuda()

        for j in range(batch_size):
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]

            cur_utt_output = cur_utt_fired_state[cur_utt_fired_mark == 1]
            cur_utt_length = cur_utt_output.size(0)

            pad_length = fired_max_length - cur_utt_length

            cur_utt_output = torch.cat([cur_utt_output,
                                        torch.full([pad_length, hidden_size], 0.0).cuda()], 0)
            cur_utt_output = cur_utt_output.unsqueeze(0)
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)
        
        # calculate the not_padding according to the cif_outputs
        not_padding_after_cif = (torch.sum(torch.abs(cif_outputs), -1) != 0.0).int()
        cif_outputs = self.normalize_scalar * F.normalize(cif_outputs, p=2, dim=-1)

        return cif_outputs, ori_weight, fired_flag, not_padding_after_cif, sum_a

class SEQSCD(Model):
    """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

    Parameters
    ----------
    sincnet : `dict`, optional
        SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
        default parameters. Use {'skip': True} to use handcrafted features
        instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
        becomes [ features -> RNN -> ...].
    rnn : `dict`, optional
        Recurrent network parameters. Defaults to `RNN` default parameters.
    ff : `dict`, optional
        Feed-forward layers parameters. Defaults to `FF` default parameters.
    """

    @staticmethod
    def get_alignment(task: Task, sincnet=None, **kwargs):
        """Get frame alignment"""

        if sincnet is None:
            sincnet = dict()
        return "center"

    @staticmethod
    def get_resolution(
        task: Task,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        **kwargs,
    ) -> Resolution:
        """Get sliding window used for feature extraction

        Parameters
        ----------
        task : Task
        sincnet : dict, optional
        rnn : dict, optional

        Returns
        -------
        sliding_window : `pyannote.core.SlidingWindow` or {`window`, `frame`}
            Returns RESOLUTION_CHUNK if model returns one vector per input
            chunk, RESOLUTION_FRAME if model returns one vector per input
            frame, and specific sliding window otherwise.
        """

        if rnn is None:
            rnn = {"pool": None}

        if rnn.get("pool", None) is not None:
            return RESOLUTION_CHUNK

        return RESOLUTION_FRAME


    def init(
        self,
        sincnet: Optional[dict] = None,
        rnn: Optional[dict] = None,
        cif: Optional[dict] = None,
        loss_cfg: Optional[dict] = None,
        training: bool = True,
    ):
        """waveform -> SincNet -> RNN [-> merge] [-> time_pool] -> FC -> output

        Parameters
        ----------
        sincnet : `dict`, optional
            SincNet parameters. Defaults to `pyannote.audio.models.sincnet.SincNet`
            default parameters. Use {'skip': True} to use handcrafted features
            instead of waveforms: [ waveform -> SincNet -> RNN -> ... ] then
            becomes [ features -> RNN -> ...].
        rnn : `dict`, optional
            Recurrent network parameters. Defaults to `RNN` default parameters.
        """

        if sincnet is None:
            sincnet = dict()
        self.sincnet = sincnet
        self.loss_cfg = loss_cfg
        self.down_rate = loss_cfg["down_rate"]
        self.rnn = rnn
        self.training = training
        self.normalize_scalar = cif["normalize_scalar"]
        n_features = self.n_features
        
        if self.rnn is None:
            self.rnn = dict()

        self.rnn_ = Encoder(n_features, self.down_rate, **rnn)
        n_features = self.rnn_.dimension

        cif['encoder_out_dim'] = n_features
        self.cif = cif
        self.cif_ = CIF(**cif)

        if self.rnn['bidirectional'] == True:
            hidden_size = self.rnn['hidden_size'] * 2
        else:
            hidden_size = self.rnn['hidden_size']

        self.dec_linear = nn.Linear(hidden_size, self.rnn['hidden_size'], bias=True)
        self.activation_hid = torch.nn.ReLU()

        num_class = loss_cfg["num_spk_class"]
        self.activation_ = torch.nn.Sigmoid()
        self.linear_ = nn.Linear(self.rnn['hidden_size'], num_class, bias=True)

    def forward(self, waveforms, mask=None):
        """Forward pass

        Parameters
        ----------
        waveforms : (batch_size, n_samples, 1) `torch.Tensor`
            Batch of waveforms. In case SincNet is skipped, a tensor with shape
            (batch_size, n_samples, n_features) is expected.
        Returns
        -------
        output : `torch.Tensor`
            Final network output.
        """
        #waveforms.size()  [64, 32000, 1]
        if self.sincnet.get("skip", False):
            output = waveforms
        else:
            output = self.sincnet_(waveforms)

        # encoder 
        output = self.rnn_(output)

        # CIF
        output, weight_keep, fired_flag, not_padding_after_cif, sum_a = self.cif_(output, mask, self.training)
        weight_keep = weight_keep.unsqueeze(-1)
        cif_output = output

        # Decoder
        # hidden FC layer 
        output = self.dec_linear(output)
        output = self.activation_hid(output)

        # output FC layer
        output = self.linear_(output)
        output = self.activation_(output)

        return cif_output, output, not_padding_after_cif, sum_a, torch.cat([1-fired_flag, fired_flag], -1), self.loss_cfg

    @property
    def dimension(self):
        return Model.dimension.fget(self)

    def intermediate_dimension(self, layer):
        if layer == 0:
            return self.sincnet_.dimension
        return self.rnn_.intermediate_dimension(layer - 1)

