# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Defines the TransformerClassifier model, and its encoder and decoder stacks.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
TransformerClassifier model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.transformer_classifier.model import attention_layer
from official.transformer_classifier.model import embedding_layer
from official.transformer_classifier.model import ffn_layer
from official.transformer_classifier.model import model_utils

_NEG_INF = -1e9


class TransformerClassifier(object):
    """TransformerClassifier model for sequence to class.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The TransformerClassifier model consists of an encoder and single
    step decoder. The input is an int sequence (or a batch of sequences).
    The encoder produces a continous representation, and then uses the
    single step decoder to find probabilities for each class.
    """

    def __init__(self, params, mode):
        """Initialize layers to build TransformerClassifier model.

        Args:
          params: hyperparameter object defining layer sizes, dropout values, etc.
          train: boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.mode = mode
        self.train = mode == tf.estimator.ModeKeys.TRAIN
        self.params = params

        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            params["vocab_size"], params["hidden_size"],
            method="matmul" if params["tpu"] else "gather")
        self.encoder_stack = EncoderStack(params, self.train)
        self.decoder_stack = DecoderStack(params, self.train)
        self.classification_network = ffn_layer.FeedForwardNetwork(
            params["num_classes"], params["filter_size"],
            params["relu_dropout"], self.train, False)

    def __call__(self, inputs):
        """Calculate class logits of the input sequences.

        Args:
          inputs: int tensor with shape [batch_size, length].

        Returns:
          If not in prediction mode, then return logits for each possible class.
          float tensor with shape [batch_size, num_classes]
          If in prediction mode, then return probability for each possible class.
            returns a dictionary {
              class_ids: [batch_size]
              prob: [batch_size, num_classes]
              logit: [batch_size, num_classes]}
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = tf.variance_scaling_initializer(
            self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
        with tf.variable_scope("TransformerClassifier", initializer=initializer):
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(inputs)

            # Run the inputs through the encoder layer to map the symbol
            # representations to continuous representations.
            encoder_outputs = self.encode(inputs, attention_bias)

            # Return logits if training is true, or return prediction otherwise.
            if self.mode == tf.estimator.ModeKeys.PREDICT:
                return self.predict(encoder_outputs, attention_bias)
            else:
                if self.mode == tf.estimator.ModeKeys.TRAIN:
                    current_checkpoint = tf.train.latest_checkpoint(self.params['model_dir'])
                    if current_checkpoint is None and self.params['transfer_model_dir'] is not None:
                        transfer_checkpoint = tf.train.latest_checkpoint(self.params['transfer_model_dir'])
                        if transfer_checkpoint is not None:
                            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model')
                            variables = [var.name.split(':')[0] for var in variables]
                            assignment_map = {_transfer_mapping(var): var for var in variables}
                            tf.train.init_from_checkpoint(transfer_checkpoint, assignment_map)
                return self.decode(encoder_outputs, attention_bias)

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        with tf.name_scope("encode"):
            # Prepare inputs to the layer stack by adding positional encodings and
            # applying dropout.
            embedded_inputs = self.embedding_softmax_layer(inputs)
            inputs_padding = model_utils.get_padding(inputs)

            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(embedded_inputs)[1]
                pos_encoding = model_utils.get_position_encoding(
                    length, self.params["hidden_size"])
                encoder_inputs = embedded_inputs + pos_encoding

            if self.train:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, 1 - self.params["layer_postprocess_dropout"])

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, encoder_outputs, attention_bias):
        """Generate logits for each class by decoding for 1 step.

        Args:
          encoder_outputs: continuous representation of input sequence.
            float tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float32 tensor with shape [batch_size, num_classes]
        """
        with tf.name_scope("decode"):
            # Prepare inputs to decoder layers by adding positional encoding.
            batch_size = tf.shape(encoder_outputs)[0]
            decoder_inputs = tf.zeros([batch_size, 1, self.params["hidden_size"]])
            with tf.name_scope("add_pos_encoding"):
                length = tf.shape(decoder_inputs)[1]    # length is 1.
                decoder_inputs += model_utils.get_position_encoding(
                    length, self.params["hidden_size"])

            # Run values
            decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
                length)
            outputs = self.decoder_stack(
                decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias)
            outputs = tf.squeeze(outputs, axis=[1])
            logits = self.classification_network(outputs)
            return logits

    def predict(self, encoder_outputs, attention_bias):
        """Return class probs and logits."""

        # Create logits of each class with shape [batch_size, num_classes]
        logits = self.decode(encoder_outputs, attention_bias)

        return {"class_ids": tf.argmax(logits, axis=1),
                "probs": tf.nn.softmax(logits),
                "logits": logits}


class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def compute_output_shape(self, input_shape):
        raise NotImplementedError("compute_output_shape is not implemented yet!")

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, _):
        self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size],
                                     initializer=tf.ones_initializer())
        self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(object):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def __init__(self, layer, params, train):
        self.layer = layer
        self.postprocess_dropout = params["layer_postprocess_dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(params["hidden_size"])

    def __call__(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = tf.nn.dropout(y, 1 - self.postprocess_dropout)
        return x + y


class EncoderStack(tf.layers.Layer):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def compute_output_shape(self, input_shape):
        raise NotImplementedError("compute_output_shape is not implemented yet!")

    def __init__(self, params, train):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(params["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], train, params["allow_ffn_pad"])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with tf.variable_scope("layer_%d" % n):
                with tf.variable_scope("self_attention"):
                    encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
                with tf.variable_scope("ffn"):
                    encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.layers.Layer):
    """Transformer decoder stack.

    Unlike the encoder stack, the decoder stack is made up of only 1 layer.
    The layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def compute_output_shape(self, input_shape):
        raise NotImplementedError("compute_output_shape is not implemented yet!")

    def __init__(self, params, train):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(1):
            self_attention_layer = attention_layer.SelfAttention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)
            enc_dec_attention_layer = attention_layer.Attention(
                params["hidden_size"], params["num_heads"],
                params["attention_dropout"], train)
            feed_forward_network = ffn_layer.FeedForwardNetwork(
                params["hidden_size"], params["filter_size"],
                params["relu_dropout"], train, params["allow_ffn_pad"])

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        self.output_normalization = LayerNormalization(params["hidden_size"])

    def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
             attention_bias, cache=None):
        """Return the output of the decoder layer stacks.

        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer.
            [1, 1, target_len, target_length]
          attention_bias: bias for encoder-decoder attention layer.
            [batch_size, 1, 1, input_length]
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
               ...}

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with tf.variable_scope(layer_name):
                with tf.variable_scope("self_attention"):
                    decoder_inputs = self_attention_layer(
                        decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
                with tf.variable_scope("encdec_attention"):
                    decoder_inputs = enc_dec_attention_layer(
                        decoder_inputs, encoder_outputs, attention_bias)
                with tf.variable_scope("ffn"):
                    decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)


def _transfer_mapping(variable_name):
    variable_name = variable_name.replace('TransformerClassifier', 'TransformerDecoder')
    variable_name = variable_name.replace('encoder', 'decoder')
    return variable_name
