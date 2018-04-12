# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------

'''
@author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
'''

import os

import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

from DLPythonNetwork import DLPythonNetwork
from DLPythonNetwork import DLPythonNetworkReader
from DLPythonNetwork import DLPythonNetworkSpec
from DLPythonNetwork import DLPythonTensorSpec

from TFModel import TFModel


class TFNetworkReader(DLPythonNetworkReader):

    def read(self, path, **kwargs):
        # Parse the SavedModel
        saved_model = self._parse_saved_model(path)
        if len(saved_model.meta_graphs) > 1:
            raise ValueError("The SavedModel must contain only one graph.")
        meta_graph = saved_model.meta_graphs[0]
        tags = meta_graph.meta_info_def.tags

        # Read the SavedModel into a graph
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess, tags, path)

        # Get the signature
        if len(meta_graph.signature_def.items()) > 1:
            raise ValueError("The SavedModel must contain only one signature.")
        signature_key, sig = list(meta_graph.signature_def.items())[0]
        method_name = sig.method_name

        # Get the input and output tensors
        get_tensor = lambda t: tf.saved_model.utils.get_tensor_from_tensor_info(t, graph=graph)
        inps = { k: get_tensor(v) for k,v in sig.inputs.items() }
        oups = { k: get_tensor(v) for k,v in sig.outputs.items() }

        return TFNetwork(TFModel(graph, inputs=inps, outputs=oups, tags=tags,
                                 method_name=method_name, signature_key=signature_key))

    # TODO! Copied from https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/saved_model/loader_impl.py
    def _parse_saved_model(self, export_dir):
        """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

        Args:
            export_dir: Directory containing the SavedModel file.

        Returns:
            A `SavedModel` protocol buffer.

        Raises:
            IOError: If the file does not exist, or cannot be successfully parsed.
        """
        # Build the path to the SavedModel in pbtxt format.
        path_to_pbtxt = os.path.join(
            tf.compat.as_bytes(export_dir),
            tf.compat.as_bytes(tf.saved_model.constants.SAVED_MODEL_FILENAME_PBTXT))
        # Build the path to the SavedModel in pb format.
        path_to_pb = os.path.join(
            tf.compat.as_bytes(export_dir),
            tf.compat.as_bytes(tf.saved_model.constants.SAVED_MODEL_FILENAME_PB))

        # Parse the SavedModel protocol buffer.
        saved_model = saved_model_pb2.SavedModel()
        if os.path.isfile(path_to_pb):
            try:
                with open(path_to_pb, 'rb') as pb_file:
                    file_content = pb_file.read()
                saved_model.ParseFromString(file_content)
                return saved_model
            except message.DecodeError as e:
                raise IOError("Cannot parse file %s: %s." % (path_to_pb, str(e)))
        elif os.path.isfile(path_to_pbtxt):
            try:
                with open(path_to_pbtxt, 'rb') as pb_file:
                    file_content = pb_file.read()
                text_format.Merge(file_content.decode("utf-8"), saved_model)
                return saved_model
            except text_format.ParseError as e:
                raise IOError("Cannot parse file %s: %s." % (path_to_pbtxt, str(e)))
        else:
            raise IOError("SavedModel file does not exist at: %s/{%s|%s}" %
                          (export_dir,
                           constants.SAVED_MODEL_FILENAME_PBTXT,
                           constants.SAVED_MODEL_FILENAME_PB))


class TFNetwork(DLPythonNetwork):

    def __init__(self, model):
        super().__init__(model)

    @property
    def spec(self):
        if self._spec is None:
            model = self._model

            inp_specs = [ self._tensor_spec(t, n) for n, t in model.inputs.items() ]
            hid_specs = [] # TODO hidden layer spec
            oup_specs = [ self._tensor_spec(t, n) for n, t in model.outputs.items() ]

            self._spec = TFNetworkSpec(inp_specs, hid_specs, oup_specs, model.tags)
        return self._spec

    def execute(self, in_data, batch_size):
        # TODO implement
        raise NotImplementedError()

    def save(self, path):
        model = self._model
        model.save(path)

    def _tensor_spec(self, tf_tensor, name=None):
        id = tf_tensor.name
        if name is None:
            name = tf_tensor.name
        tf_shape = tf_tensor.get_shape().as_list()
        if len(tf_shape) < 2:
            tf_shape.append(1)
        batch_size = tf_shape[0]
        shape = tf_shape[1:]
        element_type = tf_tensor.dtype.name
        try:
            data_format = tf_tensor.op.get_attr('data_format')
            dimension_order = 'TCDHW' if data_format.startswith("NC") else 'TDHWC'
        except ValueError:
            # Default to TDHWC TODO is this a good idea?
            dimension_order = 'TDHWC'
        return DLPythonTensorSpec(id, name, batch_size, shape, element_type, dimension_order)


class TFNetworkSpec(DLPythonNetworkSpec):

    def __init__(self, input_specs, intermediate_output_specs, output_specs, tags):
        super().__init__(input_specs, intermediate_output_specs, output_specs)
        self._tags = tags
        # TODO enforce the training config attribute in the super class?
        self.training_config = None

    @property
    def network_type(self):
        from TFNetworkType import instance as TensorFlow
        return TensorFlow()
