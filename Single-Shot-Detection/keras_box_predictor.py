import collections

import tensorflow as tf

from object_detection.core import box_predictor
from object_detection.utils import static_shape

keras = tf.keras.layers

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class _NoopVariableScope(object):
  
  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(box_predictor.KerasBoxPredictor):
 

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_heads,
               class_prediction_heads,
               other_heads,
               conv_hyperparams,
               num_layers_before_predictor,
               min_depth,
               max_depth,
               freeze_batchnorm,
               inplace_batchnorm_update,
               name=None):
 
    super(ConvolutionalBoxPredictor, self).__init__(
        is_training, num_classes, freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        name=name)
    if min_depth > max_depth:
      raise ValueError('min_depth should be less than or equal to max_depth')
    if len(box_prediction_heads) != len(class_prediction_heads):
      raise ValueError('All lists of heads must be the same length.')
    for other_head_list in other_heads.values():
      if len(box_prediction_heads) != len(other_head_list):
        raise ValueError('All lists of heads must be the same length.')

    self._prediction_heads = {
        BOX_ENCODINGS: box_prediction_heads,
        CLASS_PREDICTIONS_WITH_BACKGROUND: class_prediction_heads,
    }

    if other_heads:
      self._prediction_heads.update(other_heads)

    # We generate a consistent ordering for the prediction head names,
    # So that all workers build the model in the exact same order
    self._sorted_head_names = sorted(self._prediction_heads.keys())

    self._conv_hyperparams = conv_hyperparams
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor

    self._shared_nets = []

  def build(self, input_shapes):
    """Creates the variables of the layer."""
    if len(input_shapes) != len(self._prediction_heads[BOX_ENCODINGS]):
      raise ValueError('This box predictor was constructed with %d heads,'
                       'but there are %d inputs.' %
                       (len(self._prediction_heads[BOX_ENCODINGS]),
                        len(input_shapes)))
    for stack_index, input_shape in enumerate(input_shapes):
      net = []

      # Add additional conv layers before the class predictor.
      features_depth = static_shape.get_depth(input_shape)
      depth = max(min(features_depth, self._max_depth), self._min_depth)
      tf.logging.info(
          'depth of additional conv before box predictor: {}'.format(depth))

      if depth > 0 and self._num_layers_before_predictor > 0:
        for i in range(self._num_layers_before_predictor):
          net.append(keras.Conv2D(depth, [1, 1],
                                  name='SharedConvolutions_%d/Conv2d_%d_1x1_%d'
                                  % (stack_index, i, depth),
                                  padding='SAME',
                                  **self._conv_hyperparams.params()))
          net.append(self._conv_hyperparams.build_batch_norm(
              training=(self._is_training and not self._freeze_batchnorm),
              name='SharedConvolutions_%d/Conv2d_%d_1x1_%d_norm'
              % (stack_index, i, depth)))
          net.append(self._conv_hyperparams.build_activation_layer(
              name='SharedConvolutions_%d/Conv2d_%d_1x1_%d_activation'
              % (stack_index, i, depth),
          ))
      # Until certain bugs are fixed in checkpointable lists,
      # this net must be appended only once it's been filled with layers
      self._shared_nets.append(net)
    self.built = True

  def _predict(self, image_features):
    
    predictions = collections.defaultdict(list)

    for (index, net) in enumerate(image_features):

      # Apply shared conv layers before the head predictors.
      for layer in self._shared_nets[index]:
        net = layer(net)

      for head_name in self._sorted_head_names:
        head_obj = self._prediction_heads[head_name][index]
        prediction = head_obj(net)
        predictions[head_name].append(prediction)

    return predictions
