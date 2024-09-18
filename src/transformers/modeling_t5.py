# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """
""" Modified by NCI team """
import copy
import math
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.nn import CrossEntropyLoss

from .configuration_t5 import T5Config
from .file_utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from .modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, Seq2SeqLMOutput, Seq2SeqModelOutput
from .modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from .utils import logging
from einops import rearrange, repeat, reduce

# logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contrains shortcut names and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            # elif scope_names[0] == 'scale':
            #     pointer = getattr(pointer, 'weight')
            # elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
            #     pointer = getattr(pointer, 'bias')
            # elif scope_names[0] == 'squad':
            #     pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape, f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    # logger.info("Weights not copied to PyTorch model: {}".format(', '.join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Construct a layernorm module in the T5 style
        No bias and no substraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        # layer norm should always be calculated in float32
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)

        if self.weight.dtype == torch.float16:
            x = x.to(torch.float16)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, is_bidirectional=False):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """Compute binned relative position bias"""
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.is_bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert len(past_key_value) == 2, "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_qlen = qlen + past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """projection"""
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """compute context"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value is not None:
            if kv is None:
                k_, v_ = past_key_value
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(q, k.transpose(3, 2))  # equivalent of torch.einsum("bnqd,bnkd->bnqk", q, k), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -qlen:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=not config.is_decoder)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=True)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values,
                "2 (past / key) for cross attention" if expected_num_past_key_values == 4 else "",
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        if self.is_decoder and encoder_hidden_states is not None:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        # Add attentions if we output them
        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            d_kv = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * d_kv) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * d_kv) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.decoder_skipping = config.decoder_skipping
        self.block = nn.ModuleList([T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        token_no=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs_nocache = layer_module(
            #     hidden_states,
            #     attention_mask=extended_attention_mask,
            #     position_bias=position_bias,
            #     encoder_hidden_states=encoder_hidden_states,
            #     encoder_attention_mask=encoder_extended_attention_mask,
            #     encoder_decoder_position_bias=encoder_decoder_position_bias,
            #     head_mask=head_mask[i],
            #     past_key_value=None,
            #     use_cache=use_cache,
            #     output_attentions=output_attentions,
            # )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.
    It's an encoder decoder transformer pre-trained in a text-to-text denoising generative setting.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            T5 is a model with relative position embeddings so you should be able to pad the inputs on both the right
            and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for detail.

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a
            `T5 Training <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for sequence to sequence training. T5 uses the :obj:`pad_token_id` as the starting token for
            :obj:`decoder_input_ids` generation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at
            `T5 Training <./t5.html#training>`__. If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_input_ids` takes the value of :obj:`input_ids`.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`: `attentions`)
            :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a sequence of
            hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation.
            If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds` have to be input
            (see :obj:`past_key_values`).
            This is useful if you want more control over how to convert :obj:`decoder_input_ids` indices into
            associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both
            unset, :obj:`decoder_inputs_embeds` takes the value of :obj:`inputs_embeds`.

        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-stateswithout any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)

            >>> last_hidden_states = outputs.last_hidden_state
        """
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    authorized_missing_keys = [r"encoder\.embed_tokens\.weight", r"decoder\.embed_tokens\.weight", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        config.decoder_skipping = getattr(config, "decoder_skipping", 0)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        addtion_config_list = [
            "semantic_identifier",
            "hierarchic_decode",
            "decode_vocab_size",
            "tie_decode_embedding",
            "adaptor_decode",
            "adaptor_efficient",
            "denoising",
            "multiple_decoder",
            "decoder_num",
            "max_output_length",
            "output_vocab_size",
            "per_vocab_length",
            "Rdrop",
            "Rdrop_only_decoder",
            "Rdrop_loss",
            "embedding_distillation",
            "adaptor_dis",
            "decoder_skipping",
            "valid_logits",
            "lastlayer_lmhead",
            "use_skip_logits",
            "logits_fixer",
        ]
        for key in addtion_config_list:
            setattr(self, key, getattr(config, key, None))
        config.decoder_skipping = self.decoder_skipping
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        decoder_config.semantic_identifier = self.semantic_identifier

        if self.semantic_identifier:  # 2 by default -M
            assert config.decode_vocab_size is not None
            self.decode_embeddings = nn.Embedding(config.decode_vocab_size, config.d_model)  # 302,125
        else:
            self.decode_embeddings = self.shared

        # logger.info(f"adaptor_decode, {self.adaptor_decode == 1}")
        # logger.info(f"adaptor_efficient, {self.adaptor_efficient == 1}")
        if self.adaptor_decode and not self.adaptor_efficient:
            logger.info("adaptor without efficient")
            self.adaptor_embeddings = nn.Embedding(config.decode_vocab_size, config.d_model)
            self.adaptor = T5Stack(decoder_config, self.adaptor_embeddings)  # [batch_size, seq_len, emb_dim]
            self.adaptor_linear = nn.Linear(config.d_model, config.d_model**2, bias=False)
        elif self.adaptor_efficient:  # thisway by default -M
            logger.info("efficient adaptor")
            self.adaptor_embeddings = nn.Parameter(torch.rand(1, 1, config.d_model))
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.d_model, nhead=8)
            self.adaptor = nn.TransformerDecoder(decoder_layer, num_layers=config.adaptor_layer_num)
            self.adaptor_linear = nn.Linear(config.d_model, config.d_model * config.decode_vocab_size, bias=False)
        else:
            logger.info("deactivate adaptor")
            self.adaptor_embeddings = None
            self.adaptor = None
            self.adaptor_linear = None

        if self.semantic_identifier:  # Yes -M
            self.lm_head = nn.Linear(config.d_model, config.decode_vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        ### original vocab_size = 32128, decode_vocab_size(2)=302

        t5stack_module = T5Stack if not self.decoder_skipping else T5StackSkipping
        self.decoder = t5stack_module(decoder_config, self.decode_embeddings)  # [batch_size, seq_len, emb_dim]

        if self.tie_decode_embedding:  # Yes
            self._tie_or_clone_weights(self.lm_head, self.decode_embeddings)  # [output_embedding , input_embedding] -M XXX: wait, that?

        if self.adaptor_dis:
            self.distill_adaptor = nn.Linear(config.d_model, config.decode_vocab_size, bias=False)

        if self.semantic_identifier == 2:
            # init decoder valid mask
            bz = 1
            seq_length = config.max_output_length
            vocab_size = config.decode_vocab_size

            output_vocab_size = config.output_vocab_size
            logger.info(f"bz {bz}, seq_length {seq_length}, vocab_size {vocab_size}, output_vocab_size {output_vocab_size}")
            valid_indices = torch.arange(output_vocab_size).view(1, -1)
            pos_indices = torch.arange(seq_length).view(-1, 1) * output_vocab_size

            # valid_indices = torch.arange(10).view(1, -1)
            # pos_indices = torch.arange(seq_length).view(-1, 1) * 10
            valid_indices = valid_indices + pos_indices + 2  # [seq_length, 10]
            ones_indices = torch.ones(seq_length, 1).to(valid_indices.device)
            valid_indices = torch.cat((valid_indices, ones_indices), dim=-1).long()
            self.valid_indices = valid_indices.sort()[0]

            valid_indices[-1, :] = torch.ones(1, output_vocab_size + 1)
            # valid_indices[-1,:] = torch.ones(1, 11)
            valid_indices = valid_indices.unsqueeze(0).repeat([1, 1, 1])  # [bz, 10, sl]
            zero_mask = torch.zeros(1, seq_length, vocab_size)
            mask = zero_mask - 1e9
            self.logit_mask = mask.scatter_(-1, valid_indices, zero_mask)  # scatter explained: https://zhuanlan.zhihu.com/p/339043454
        else:
            self.valid_indices = None
            self.logit_mask = None

        if self.adaptor_dis:
            self.distill_adaptor = nn.Linear(config.d_model, config.decode_vocab_size, bias=False)

        if self.adaptor_efficient and config.adaptor_layer_num > 0:
            self.apply_adaptor = self.apply_original_adaptor
        else:
            self.apply_adaptor = self.apply_lm_head
        self.init_weights()

    def apply_lm_head(self, decoder_input_ids, sequence_output):
        return self.lm_head(sequence_output)

    def apply_mutilple_lm_head(self, decoder_input_ids, sequence_output, lm_head_number=None):  # Deprecated
        # sequence_output: [batch_size,seq_length, config.d_model]
        # self.lm_head_params: [seq_length, config.d_model, config.decode_vocab_size]
        if lm_head_number is None:
            return torch.matmul(sequence_output.unsqueeze(-2), self.lm_head_params).squeeze(-2)
        else:
            # return torch.matmul(sequence_output[:,-1,:].unsqueeze(1), self.lm_head_params[lm_head_number])
            return torch.matmul(sequence_output.unsqueeze(-2), self.lm_head_params[: lm_head_number + 1]).squeeze(-2)

    def apply_distilled_adaptor(self, decoder_input_ids, sequence_output):
        return self.distill_adaptor(sequence_output)  # + self.lm_head(sequence_output)

    def apply_original_adaptor(self, decoder_input_ids, sequence_output):
        lm_head_weight = self.lm_head.weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, config.d_model, vocab_size]
        # logger.info("decoder_input_ids",decoder_input_ids)
        decoder_input_embedding = self.decode_embeddings(decoder_input_ids)  # [batch_size, seq_length, config.d_model]
        batch_size = decoder_input_ids.shape[0]
        seq_length = decoder_input_embedding.shape[1]

        def generate_square_subsequent_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            return mask

        mask = generate_square_subsequent_mask(seq_length).to(decoder_input_embedding.device)
        encode_embedding = self.adaptor_embeddings + torch.zeros(batch_size, 1, 1).to(decoder_input_embedding.device)
        decoder_input_embedding = self.adaptor(decoder_input_embedding.transpose(0, 1), encode_embedding.transpose(0, 1), tgt_mask=mask).transpose(
            0, 1
        )
        # [batch_size, seq_length, config.d_model]
        adaptor_weight = self.adaptor_linear(decoder_input_embedding).reshape(
            decoder_input_embedding.shape[0], decoder_input_embedding.shape[1], self.model_dim, -1
        )
        # [batch_size, seq_length, config.d_model, vocab_size]
        lm_head_weight = adaptor_weight + lm_head_weight
        lm_logits = torch.matmul(sequence_output.unsqueeze(-2), lm_head_weight)
        lm_logits = lm_logits.squeeze(-2)
        lm_head_weight = self.lm_head.weight.T.unsqueeze(0).unsqueeze(0)  # [1, 1, config.d_model, vocab_size]
        return lm_logits

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_callable(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        input_mask=None,
        logit_mask=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        only_encoder=False,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        lm_weights=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        exit_analysis=None,
        token_no=None,
        training_mode=True,
        max_position=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """

        if "lm_labels" in kwargs:
            # warnings.warn(
            #     "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
            #     FutureWarning,
            # )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        training_mode = False if not self.training else training_mode  # Allow external override of execution modes while training -M
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if past_key_values is not None:
            assert token_no is not None, f"should specifc token number when has past_key_values"
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:  # Passed
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:  # Passed
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids_full = decoder_input_ids
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds_full = decoder_inputs_embeds
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
            else:
                decoder_inputs_embeds_full = None

        # Decode

        if self.decoder_skipping:
            kwargs = {
                "exit_analysis": exit_analysis,
                "lm_head": self.lm_head if self.config.classifier_type == "last_layer_softmax" else None,
                "training_mode": training_mode,
                "max_position": max_position,
            }
        else:
            kwargs = {}

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_no=token_no,
            **kwargs,
        )
        # if past_key_values is not None:
        #     decoder_outputs_nocache = self.decoder(
        #         input_ids=decoder_input_ids_full,
        #         attention_mask=decoder_attention_mask,
        #         inputs_embeds=decoder_inputs_embeds_full,
        #         past_key_values=None,
        #         encoder_hidden_states=hidden_states,
        #         encoder_attention_mask=attention_mask,
        #         head_mask=head_mask,
        #         use_cache=False,
        #         output_attentions=output_attentions,
        #         output_hidden_states=output_hidden_states,
        #         return_dict=return_dict,
        #         **kwargs,
        #     )
        #     print("decoder_outputs_nocache", decoder_outputs_nocache[0].shape)
        #     print((decoder_outputs_nocache.last_hidden_state[:, -1, :] - decoder_outputs.last_hidden_state[:, -1, :]).abs().sum())
        #     # assert decoder_outputs_nocache.last_hidden_state[0][-1].allclose(decoder_outputs.last_hidden_state[0])
        #     pass

        sequence_output = decoder_outputs[0]
        # logger.info("sequence_output", sequence_output.shape)

        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)  # shape(batch_size, sequence_length, config.d_model)

        if training_mode:  # TODO: Unify the training and inference code
            # apply adaptor and lm head
            if self.adaptor_efficient:
                lm_logits = self.apply_original_adaptor(decoder_input_ids, sequence_output)  # QUESTION: learn the EOS token?
                if self.adaptor_dis:
                    distilled_logits = self.apply_distilled_adaptor(decoder_input_ids, sequence_output)
                    distilled_logits = distilled_logits + self.logit_mask.to(lm_logits.device)
            else:
                lm_logits = self.lm_head(sequence_output)
            lm_logits = lm_logits.float()
            logits_before_mask = lm_logits.clone()
            # pure_logits = (
            #     select_valid_embedding(pure_logits, mode="ones") if (self.valid_logits >= 2) and self.semantic_identifier == 2 else pure_logits
            # )
            ### valid_logits: 0 - no selection, 1 - attention mask only, 2 - valid and attention mask , 3 - valid
            if self.semantic_identifier == 2:
                lm_logits += self.logit_mask.to(lm_logits.device)
        else:  # inference
            if self.decoder_skipping and self.use_skip_logits:
                skip_outputs = decoder_outputs.skip_outputs
                skip_logits, skipped = skip_outputs["skip_logits"], skip_outputs["skipped"]
                if exit_analysis:
                    final_lm_logits = self.lm_head(sequence_output[:, -1:, :])
                    skip_logits[~skipped] = final_lm_logits[~skipped]
                else:
                    lm_logits_unskipped = self.lm_head(sequence_output[~skipped, -1:, :])
                    skip_logits[~skipped] = lm_logits_unskipped
                lm_logits = skip_logits
            elif self.decoder_skipping:
                raise NotImplementedError
            else:
                lm_logits = self.lm_head(sequence_output[:, -1:, :])

            if self.semantic_identifier == 2:
                # mask = self.logit_mask[:, : lm_logits.shape[1], :] if past_key_values is None else self.logit_mask[:, token_no, :].unsqueeze(1)
                mask = self.logit_mask[:, : lm_logits.shape[1], :] if training_mode else self.logit_mask[:, token_no, :].unsqueeze(1)
                lm_logits += mask.expand((lm_logits.shape[0], -1, -1)).to(lm_logits.device)

        loss = 0.0
        if labels is not None and not self.decoder_skipping:
            if self.Rdrop > 0 and training_mode:
                bz = lm_logits.shape[0]
                sl = lm_logits.shape[1]
                orig_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                neg_logits_1 = sequence_output.transpose(0, 1)  # [sl, bz, vocab_size]
                neg_logits_2 = neg_logits_1.transpose(1, 2)  # [sl, vocab_size, bz]
                neg_logits = torch.bmm(neg_logits_1, neg_logits_2)  # [sl, bz, bz_logits]
                neg_mask = -1e9 * torch.eye(bz).to(neg_logits.device)
                neg_logits = neg_logits + neg_mask.unsqueeze(0)
                neg_logits = F.softmax(neg_logits.view(-1, bz), dim=-1)  # [sl*bz, bz_logits]
                contrast_labels = torch.cat([torch.arange(bz // 2, bz), torch.arange(0, bz // 2)], dim=-1)
                contrast_labels = contrast_labels.to(neg_logits.device).long()
                contrast_labels = contrast_labels.unsqueeze(0).repeat(sl, 1).view(-1)
                dist_loss = loss_fct(neg_logits, contrast_labels)
                loss = orig_loss + self.Rdrop * dist_loss
            else:
                orig_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                loss = orig_loss
        elif self.decoder_skipping and labels is not None:
            loss = torch.tensor(0.0).to(lm_logits.device)
            orig_loss = 0.0
            dist_loss = 0.0
        else:
            loss = None
            orig_loss = 0.0
            logits_before_mask = None
            dist_loss = 0.0

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        to_scalar = lambda x: x.detach().item() if type(x) == torch.Tensor else x

        return_result = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return_result.labels = labels
        return_result.orig_loss = to_scalar(orig_loss)
        return_result.encoder_outputs = encoder_outputs
        return_result.lm_logits = lm_logits
        if self.decoder_skipping:  # TODO: reduce redundant return, organize the return_result, split training return and inference return
            skip_position = decoder_outputs.skip_outputs["position"]
            sp_mean = (
                skip_position[decoder_attention_mask.bool()].float().mean() if decoder_attention_mask is not None else skip_position.float().mean()
            )
            return_result.skip_position = sp_mean
            # return_result.SP_per_sample= decoder_outputs.skip_outputs["position"]
            return_result.skip_outputs = decoder_outputs.skip_outputs
            if not training_mode and exit_analysis:
                return_result.final_lm_logits = final_lm_logits

        return_result.logits_before_mask = logits_before_mask
        return_result.adaptor_distillled_logits = distilled_logits if self.adaptor_dis else None
        return_result.dist_loss = self.Rdrop * to_scalar(dist_loss) if self.Rdrop > 0 else 0.0
        return return_result

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, use_cache, encoder_outputs, **kwargs):
        # cut decoder_input_ids if past is used
        token_no = input_ids.shape[1] - 1
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "token_no": token_no,
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (layer_past_state.index_select(0, beam_idx),)

            # assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            # assert len(reordered_layer_past_states) == len(layer_past_states)
            # Ignore them due to Beam Pruning

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


"""
    T5StackSkipping
    Modified from T5Stack
"""


class bias_linear(nn.Linear):
    def __init__(self, in_features, out_features, layer_nums, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.out_features = out_features
        self.bias = nn.Parameter(torch.empty(self.bias.shape[0] * layer_nums, device=self.bias.device, dtype=self.bias.dtype))
        self.reset_parameters()

    def forward(self, x, layer_id):
        # layer_id begin from 0
        x = F.linear(x, self.weight, self.bias[layer_id * self.out_features : (layer_id + 1) * self.out_features])
        return x


class LogitsFixer(nn.Module):
    def __init__(self, config, valid_indices, **kwargs):
        super().__init__()
        mid_dim = config.fixer_midlayer_dim if config.fixer_midlayer_num > 0 else config.d_model

        def mid_layer_func(first_layer=False):
            return nn.Sequential(
                nn.Linear(config.d_model, mid_dim) if first_layer else nn.Linear(mid_dim, mid_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            )

        self.layernorm = T5LayerNorm(config.d_model)
        self.droupout = nn.Dropout(0.1)
        self.model_dim_factor = config.d_model**-0.5
        self.mid_layer = nn.Sequential(*[mid_layer_func(i == 0) for i in range(config.fixer_midlayer_num)])
        self.final_linear = nn.Linear(mid_dim, config.decode_vocab_size)

    def forward(self, sequence_output, token_no):
        x = self.mid_layer(self.droupout(self.layernorm(sequence_output * self.model_dim_factor)))
        return self.final_linear(x)


class MultipleAdaptorToken(nn.Module):
    def __init__(self, module, max_token_nums, config, *args, **kwargs):
        super().__init__()
        self.module = nn.ModuleList(module(no_final=False, config=config, *args, **kwargs) for _ in range(max_token_nums))
        self.max_token_nums = max_token_nums

    def forward(self, token_no, *args, **kwargs):
        token_no = token_no if token_no < self.max_token_nums else self.max_token_nums - 1
        return self.module[token_no](*args, **kwargs)


class MultipleAdaptorLayer(nn.Module):
    def __init__(self, module, config, *args, **kwargs):
        def attr_recursion(module, namelist, attr):
            if len(namelist) != 1:
                attr_recursion(module.__getattr__(namelist[0]), namelist[1:], attr)
            else:
                module.__setattr__(namelist[0], attr)

        super().__init__()
        self.module = nn.ModuleList(module(config=config, *args, **kwargs) for _ in range(config.num_decoder_layers))
        self.num_decoder_layers = config.num_decoder_layers
        for k, v in self.module[0].named_parameters():
            for i in range(self.num_decoder_layers - 1):
                if "weight" in k and "layernorm" not in k:
                    attr_recursion(self.module[i], k.split("."), v)

    def forward(self, layer_no, *args, **kwargs):
        return self.module[layer_no](*args, **kwargs)


class LogitsFixer_HW(nn.Module):
    def __init__(self, config, no_final=False, **kwargs):
        super().__init__()
        mid_dim = config.fixer_midlayer_dim if config.fixer_midlayer_num > 0 else config.d_model
        layer_num = config.num_decoder_layers

        class mid_layer(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = bias_linear(input_dim, output_dim, layer_num)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x, layer_no):
                x = self.linear(x, layer_no)
                x = self.relu(x)
                x = self.dropout(x)
                return x

        class identity(nn.Module):
            def forward(self, x, layer_no):
                return x

        def mid_layer_func(first_layer=False):
            return mid_layer(config.d_model if first_layer else mid_dim, mid_dim)

        self.layernorm = T5LayerNorm(config.d_model)
        self.droupout = nn.Dropout(0.1)
        self.model_dim_factor = config.d_model**-0.5
        self.mid_layer = nn.ModuleList([mid_layer_func(i == 0) for i in range(config.fixer_midlayer_num)])
        if not no_final:
            self.final_linear = bias_linear(mid_dim, config.decode_vocab_size, layer_num)
        else:
            self.final_linear = identity()

    def forward(self, sequence_output, layer_no):
        x = self.droupout(self.layernorm(sequence_output * self.model_dim_factor))
        for mid_layer in self.mid_layer:
            x = mid_layer(x, layer_no)
        return self.final_linear(x, layer_no)


class LogitsFixerSingle(nn.Module):
    def __init__(self, config, no_final=False, **kwargs):
        super().__init__()
        mid_dim = config.fixer_midlayer_dim if config.fixer_midlayer_num > 0 else config.d_model

        class mid_layer(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                x = self.dropout(x)
                return x

        class identity(nn.Module):
            def forward(self, x, layer_no):
                return x

        def mid_layer_func(first_layer=False):
            return mid_layer(config.d_model if first_layer else mid_dim, mid_dim)

        self.layernorm = T5LayerNorm(config.d_model)
        self.droupout = nn.Dropout(0.1)
        self.model_dim_factor = config.d_model**-0.5
        self.mid_layer = nn.ModuleList([mid_layer_func(i == 0) for i in range(config.fixer_midlayer_num)])
        if not no_final:
            self.final_linear = nn.Linear(mid_dim, config.decode_vocab_size)
        else:
            self.final_linear = identity()

    def forward(self, sequence_output, *args, **kwargs):
        x = self.droupout(self.layernorm(sequence_output * self.model_dim_factor))
        for mid_layer in self.mid_layer:
            x = mid_layer(x)
        return self.final_linear(x)


class ExitClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = T5LayerNorm(config.d_model)
        self.droupout = nn.Dropout(0.1)
        self.model_dim_factor = config.d_model**-0.5
        self.linear = nn.Linear(config.d_model, 1)

    def forward(self, sequence_output, **args):
        x = self.droupout(self.layernorm(sequence_output * self.model_dim_factor))
        return F.linear(x)


class ExitClassifierM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = T5LayerNorm(config.d_model)
        self.droupout = nn.Dropout(0.1)
        self.model_dim_factor = config.d_model**-0.5
        self.linears = nn.ModuleList([nn.Linear(config.d_model, 1) for _ in range(config.max_output_length)])

    def train(self, mode=True):
        self.training = mode

    def forward(self, sequence_output, token_id):
        x = self.droupout(self.layernorm(sequence_output * self.model_dim_factor))
        if self.training:
            return torch.stack([self.linears[i](x[:, i, :]) for i in range(x.shape[1])], dim=1)
        else:
            return self.linears[token_id](x)


class T5StackSkipping(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList([T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.max_output_length = config.max_output_length
        self.output_vocab_size = config.output_vocab_size
        self.decode_vocab_size = config.decode_vocab_size
        self.per_vocab_length = config.per_vocab_length
        self.CALM_thresholds = config.CALM_thresholds
        self.model_dim = config.d_model
        self.classifier_type = config.classifier_type
        self.semantic_identifier = config.semantic_identifier
        self.use_skip_logits = config.use_skip_logits
        self.force_cache = config.force_cache
        if self.semantic_identifier == 2:
            valid_indices = torch.arange(config.output_vocab_size).view(1, -1)
            pos_indices = torch.arange(config.max_output_length).view(-1, 1) * config.output_vocab_size
            valid_indices = valid_indices + pos_indices + 2
            ones_indices = torch.ones(config.max_output_length, 1).to(valid_indices.device)
            self.valid_indices = torch.cat((valid_indices, ones_indices), dim=-1).long()
            self.valid_indices, _ = self.valid_indices.sort()
        else:
            self.valid_indices = torch.arange(self.decode_vocab_size * self.max_output_length).view(self.max_output_length, -1)
        self.valid_maskes = torch.zeros(self.max_output_length, self.decode_vocab_size)
        self.valid_maskes.scatter_(-1, self.valid_indices, 1)

        fixer_dict = {
            "M": LogitsFixer,
        }
        fixer_dict_block = {
            "blockA": None,
            "blockB": None,
        }
        if "single" in config.classifier_type:
            self.skip_adaptor = LogitsFixerSingle(config)
        if "HW" in config.classifier_type:
            if "multiple" in config.classifier_type:
                self.skip_adaptor = MultipleAdaptorToken(LogitsFixer_HW, 6, config)
            else:
                self.skip_adaptor = LogitsFixer_HW(config)
        logger.info(f"classifier_type:{config.classifier_type}")
        for i, layer_module in enumerate(self.block):
            layer_module.skip_params_sequence = nn.Parameter(torch.ones(config.max_output_length) * 0.1)
            # layer_module.skip_fixer = LogitsFixer(config)
            if "fixer_softmax" == config.classifier_type:
                layer_module.skip_fixer = fixer_dict[config.fixer_type](config, valid_indices=self.valid_indices)
                # layer_module.skip_linear = nn.Linear(config.d_model, config.decode_vocab_size)
            if "classifier" in config.classifier_type:
                layer_module.skip_classifier = ExitClassifierM(config)

        if "block" in config.fixer_type:
            self.skip_fixer = fixer_dict_block[config.fixer_type](config)
        self.fixer_type = config.fixer_type

        self.init_weights()
        # self.redundant = torch.tensor([12, 12, 1, 1, 0, 0, 0])
        self.redundant = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0])

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        exit_analysis=False,
        lm_head=None,
        token_no=None,
        training_mode=True,
        max_position=None,
    ):
        # output_hidden_states = True  ### TEMP ###

        exit_analysis = True if training_mode else exit_analysis

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = False if training_mode else use_cache

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        if training_mode:
            token_no = seq_length
        else:
            token_no = token_no if past_key_values is not None else seq_length - 1
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(self)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)

        ### skip part ###
        device = self.device
        # default zero values
        layer_outputs = [None] * 4
        layer_outputs[0] = torch.empty(batch_size, seq_length, self.config.d_model).to(device)
        layer_outputs[2] = torch.empty(batch_size, 16, seq_length, token_no + 1 if use_cache else seq_length).to(device)
        layer_outputs[3] = torch.empty(batch_size, 16, seq_length, 40).to(device)
        # avoid the first layer being skipped makes the shape of input deformed
        logits_length = seq_length if training_mode else 1
        skip_logits = torch.zeros(input_ids.shape[0], logits_length, self.decode_vocab_size).to(device)

        skipped = torch.tensor([False] * len(input_ids)).to(device)
        unskipped_mask = ~skipped
        skipped_at = torch.tensor([len(self.block)] * len(input_ids)).to(device)
        skipped_prediction = torch.tensor([-7] * len(input_ids)).to(device)

        if training_mode:  # expand the mask to the whole sequence while training
            skipped = skipped.unsqueeze(1).repeat(1, seq_length).to(device)
            skipped_at = skipped_at.unsqueeze(1).repeat(1, seq_length).to(device)
            skipped_prediction = skipped_prediction.unsqueeze(1).repeat(1, seq_length).to(device)

        if use_cache:
            zero_shapes = [
                (batch_size, 16, token_no + 1, 64),
                (batch_size, 16, token_no + 1, 64),
                (batch_size, 16, 40, 64),
                (batch_size, 16, 40, 64),
            ]
            zero_past_key_value = [torch.zeros(*shape).to(device) for shape in zero_shapes]
        valid_mask = self.valid_maskes[token_no] if not training_mode else self.valid_maskes
        valid_mask = valid_mask.to(device)

        if exit_analysis:  # ready to save skip data
            all_skip_score = ()
            all_logits = ()
            all_skip_desicion = ()
            all_layer_prediction = ()

        if self.classifier_type == "hs_cos":
            cosine_fun = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            last_hs = None

        if self.CALM_thresholds:
            threshold = self.block[0].skip_params_sequence[0]
            CALM_thresholds = [(9 / 10 * threshold + 0.1 * torch.exp(torch.tensor(-4 * (i) / 10))).clip(0, 1) for i in range(self.max_output_length)]
            CALM_thresholds = torch.tensor(CALM_thresholds).to(device)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_inputs = {
                "hidden_states": hidden_states,  # [batch_size, seq_length, hidden_size]
                "attention_mask": extended_attention_mask,
                "position_bias": position_bias,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_extended_attention_mask,
                "encoder_decoder_position_bias": encoder_decoder_position_bias,
                "head_mask": head_mask[i],
                "past_key_value": past_key_value,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
            }
            if training_mode or i >= self.redundant[token_no]:  # skip all calculation if redundant
                ### prepare hidden state
                if exit_analysis or not ("prophet" in self.classifier_type and skip_desicion.max() == 0):
                    hidden_states_unskiped = hidden_states[unskipped_mask] if not exit_analysis else hidden_states
                    if not training_mode and hidden_states_unskiped.size(1) > 1:
                        hidden_state = hidden_states_unskiped[:, -1:, :]
                    else:
                        hidden_state = hidden_states_unskiped
                    hidden_state = hidden_state.detach().clone()

                def calc_logits(hs_input):
                    if "last_layer_softmax" == self.classifier_type:
                        hs_input = self.dropout(self.final_layer_norm(hs_input))
                        hs_input *= self.model_dim**-0.5
                        logits = lm_head(hs_input)
                        return logits
                    if "HW" in self.classifier_type:
                        if "multiple" in self.classifier_type:
                            if training_mode:
                                # logits = torch.stack([self.skip_adaptor(token_no,hs_input[:,token_no,],i) for token_no in range(seq_length)],dim=1)
                                logits = torch.stack(
                                    [
                                        self.skip_adaptor(
                                            token_no,
                                            hs_input[
                                                :,
                                                token_no,
                                            ],
                                            i,
                                        )
                                        for token_no in range(seq_length)
                                    ],
                                    dim=1,
                                )
                            else:
                                logits = self.skip_adaptor(token_no, hs_input, i)
                        else:
                            logits = self.skip_adaptor(hs_input, i)
                    elif "single" in self.classifier_type:
                        logits = self.skip_adaptor(hs_input, i)
                    elif "fixer_softmax" == self.classifier_type:
                        logits = layer_module.skip_fixer(hs_input, token_no)
                    else:
                        raise NotImplementedError
                    return logits

                ### get clean logits for fixer_softmax
                if "softmax" in self.classifier_type or exit_analysis:  # calculate layer prediction & return
                    fixer_logits = calc_logits(hidden_state)  # need to be calculated before decision
                    if self.semantic_identifier == 2:
                        selected_logits = fixer_logits.masked_fill((1 - valid_mask).bool(), float("-inf"))
                    else:
                        selected_logits = fixer_logits
                    selected_logits = selected_logits.softmax(dim=-1)
                    top2_values, top2_indices = selected_logits.topk(2, dim=-1)

                if "softmax" in self.classifier_type:
                    score = (top2_values[:, :, 0] - top2_values[:, :, 1]).squeeze(1)
                    skip_threshold = layer_module.skip_params_sequence.to(device)
                    if self.CALM_thresholds:
                        if skip_threshold[-1] > 1:
                            skip_threshold = skip_threshold[-1]  # Do not skip
                        else:
                            skip_threshold = CALM_thresholds[token_no].to(device)
                    else:
                        skip_threshold = skip_threshold[token_no] if not training_mode else skip_threshold
                    skip_desicion = score >= skip_threshold  # if i not in [0, 11] else torch.zeros_like(score).bool()
                elif self.classifier_type == "classifier":
                    score = layer_module.skip_classifier(hidden_state, token_no).sigmoid().squeeze().squeeze()
                    skip_threshold = layer_module.skip_params_sequence.to(device)
                    skip_threshold = skip_threshold[token_no] if not training_mode else skip_threshold
                    skip_desicion = score >= skip_threshold  # if i not in [0, 11] else torch.zeros_like(score).bool()
                elif self.classifier_type == "hs_cos":
                    if last_hs is None:
                        last_hs = hidden_states if training_mode else hidden_states[:, -1, :]
                        score = torch.zeros(*last_hs.shape[: 2 if training_mode else 1], device=device)
                    else:
                        if training_mode:
                            score = torch.stack(
                                [cosine_fun(hidden_states[:, i, :], last_hs[:, i, :]) for i in range(hidden_states.shape[1])],
                                dim=1,
                            ).squeeze()
                            last_hs = hidden_states
                        else:
                            score = cosine_fun(hidden_states[:, -1, :], last_hs).squeeze()
                            last_hs = hidden_states[:, -1, :]
                    score = score.squeeze()
                    skip_threshold = layer_module.skip_params_sequence
                    skip_threshold = skip_threshold[token_no] if not training_mode else skip_threshold
                    skip_threshold = 1 - ((1 - skip_threshold) * 0.01)
                    skip_desicion = score >= skip_threshold  # if i not in [0, 11] else torch.zeros_like(score).bool()
                    # skip_desicion = torch.zeros_like(skip_desicion).bool if i == 0 else skip_desicion  # TEMP
                    skip_desicion = skip_desicion[unskipped_mask] if not exit_analysis else skip_desicion

                # Force to skip if previous layer is skipped ###TEST###
                if self.force_cache and (max_position is not None and torch.any(unskipped_mask)):
                    if exit_analysis:
                        skip_desicion[unskipped_mask] = skip_desicion[unskipped_mask] | (i >= max_position[unskipped_mask])
                    else:
                        skip_desicion = skip_desicion | (i >= max_position[unskipped_mask])

                ### dealing with outputs
                if exit_analysis:
                    layer_prediction = top2_indices[:, :, 0].squeeze(-1)
                    this_skip = (skip_desicion | skipped) ^ skipped  ### IAAI
                    skipped = skip_desicion | skipped
                    skipped_at[this_skip] = i
                    skip_logits[this_skip] = fixer_logits[this_skip]

                    all_skip_score = all_skip_score + (score,)
                    all_logits = all_logits + (fixer_logits,)
                    all_skip_desicion = all_skip_desicion + (skip_desicion,)
                    all_layer_prediction = all_layer_prediction + (layer_prediction,)

                else:
                    this_skip = torch.zeros_like(skipped).bool()
                    this_skip[unskipped_mask] = skip_desicion
                    skipped = this_skip | skipped
                    skipped_at[this_skip] = i
                    if torch.any(skip_desicion):  # at least one sample is skipped
                        skip_logits[this_skip] = calc_logits(hidden_state)[skip_desicion]

            if torch.all(skipped) and not exit_analysis:
                break

            ### calculate block output
            unskipped_mask = ~skipped
            if not exit_analysis and torch.any(skipped):  # Do not replace inputs
                for k, v in layer_inputs.items():
                    if v is not None and type(v) != bool:
                        if k == "past_key_value":
                            # if pruning_mask is not None: # For KV cache optimized, reduce the slicing
                            #     double_mask = ~pruning_mask.clone()
                            #     double_mask[double_mask.nonzero()[~unskipped_mask]] = False
                            # else:
                            #    double_mask = unskipped_mask
                            layer_inputs[k] = tuple(v[unskipped_mask] for v in layer_inputs[k])
                        else:
                            layer_inputs[k] = v[unskipped_mask]

                unskipped_layer_outputs = layer_module(**layer_inputs)

                if i == 0:
                    layer_outputs[2][unskipped_mask] = unskipped_layer_outputs[2]
                    layer_outputs[3][unskipped_mask] = unskipped_layer_outputs[3]
                #     size = unskipped_layer_outputs[0].shape
                #     layer_outputs[0] = torch.empty(self.max_output_length, *size[1:]).to(unskipped_layer_outputs[0].device)
                layer_outputs[0][unskipped_mask] = unskipped_layer_outputs[0]
                if use_cache:
                    if i > 0:
                        layer_outputs = list(layer_outputs)
                        layer_outputs[1] = [elements.clone() for elements in layer_outputs[1]]
                        for j, vv in enumerate(unskipped_layer_outputs[1]):  # past_key_value
                            layer_outputs[1][j][unskipped_mask] = vv
                    else:  # in this situation, exit happened in the first layer and the shape of past_key_value is not correct
                        for j, vv in enumerate(unskipped_layer_outputs[1]):
                            zero_past_key_value[j][unskipped_mask] = vv.clone()
                        layer_outputs[1] = zero_past_key_value
                ### We do not need to update the last 2 elements in _layer_out
            else:
                layer_outputs = layer_module(**layer_inputs)

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if use_cache:
            padding_size = len(self.block) - len(present_key_value_states)
            if len(present_key_value_states) == 0:
                present_key_value_states = tuple((zero_past_key_value,) * padding_size)
            else:
                if len(self.block) - len(present_key_value_states) > 0:
                    present_key_value_states = (
                        tuple(present_key_value_states) + (present_key_value_states[-1],) * padding_size
                    )  # Keep the same length for present_key_value_states

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_value_states, all_hidden_states, all_attentions] if v is not None)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        output.skip_outputs = {
            "skipped": skipped,
            "position": skipped_at,
            "skip_logits": skip_logits,
        }

        if exit_analysis:
            output.skip_outputs.update(
                {
                    "logits": torch.stack(all_logits, dim=2),  # [batch_size, seq_length, num_layers, vocab_size]
                    "layer_prediction": torch.stack(all_layer_prediction, dim=-1),  # [batch_size, seq_length, num_layers]
                    "skip_score": torch.stack(all_skip_score, dim=-1),  # [batch_size, seq_length, num_layers]
                    "skip_decision": torch.stack(all_skip_desicion, dim=-1),  # [batch_size, seq_length, num_layers]
                }
            )
            if output_hidden_states:
                output.skip_outputs.update({"all_hs": torch.stack(all_hidden_states).transpose(0, 1)[:, :, -1:, :]})
        return output
