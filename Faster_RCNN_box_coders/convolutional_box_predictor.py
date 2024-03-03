import functools
import tensorflow as tf
from object_detection.core import box_predictor
from object_detection.utils import static_shape

slim = tf.contrib.slim

BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = (
    box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND)
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS


class _NoopVariableScope(object):
  """A dummy class that does not push any scope."""

  def __enter__(self):
    return None

  def __exit__(self, exc_type, exc_value, traceback):
    return False


class ConvolutionalBoxPredictor(box_predictor.BoxPredictor):
 

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               num_layers_before_predictor,
               min_depth,
               max_depth):
   

 
    super(ConvolutionalBoxPredictor, self).__init__(is_training, num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._min_depth = min_depth
    self._max_depth = max_depth
    self._num_layers_before_predictor = num_layers_before_predictor

  @property
  def num_classes(self):
    return self._num_classes

  def _predict(self, image_features, num_predictions_per_location_list):
   
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    # TODO(rathodv): Come up with a better way to generate scope names
    # in box predictor once we have time to retrain all models in the zoo.
    # The following lines create scope names to be backwards compatible with the
    # existing checkpoints.
    box_predictor_scopes = [_NoopVariableScope()]
    if len(image_features) > 1:
      box_predictor_scopes = [
          tf.variable_scope('BoxPredictor_{}'.format(i))
          for i in range(len(image_features))
      ]
    for (image_feature,
         num_predictions_per_location, box_predictor_scope) in zip(
             image_features, num_predictions_per_location_list,
             box_predictor_scopes):
      net = image_feature
      with box_predictor_scope:
        with slim.arg_scope(self._conv_hyperparams_fn()):
          with slim.arg_scope([slim.dropout], is_training=self._is_training):
            # Add additional conv layers before the class predictor.
            features_depth = static_shape.get_depth(image_feature.get_shape())
            depth = max(min(features_depth, self._max_depth), self._min_depth)
            tf.logging.info('depth of additional conv before box predictor: {}'.
                            format(depth))
            if depth > 0 and self._num_layers_before_predictor > 0:
              for i in range(self._num_layers_before_predictor):
                net = slim.conv2d(
                    net,
                    depth, [1, 1],
                    reuse=tf.AUTO_REUSE,
                    scope='Conv2d_%d_1x1_%d' % (i, depth))
            sorted_keys = sorted(self._other_heads.keys())
            sorted_keys.append(BOX_ENCODINGS)
            sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
            for head_name in sorted_keys:
              if head_name == BOX_ENCODINGS:
                head_obj = self._box_prediction_head
              elif head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
                head_obj = self._class_prediction_head
              else:
                head_obj = self._other_heads[head_name]
              prediction = head_obj.predict(
                  features=net,
                  num_predictions_per_location=num_predictions_per_location)
              predictions[head_name].append(prediction)
    return predictions


# TODO(rathodv): Replace with slim.arg_scope_func_key once its available
# externally.
def _arg_scope_func_key(op):
  """Returns a key that can be used to index arg_scope dictionary."""
  return getattr(op, '_key_op', str(op))


# TODO(rathodv): Merge the implementation with ConvolutionalBoxPredictor above
# since they are very similar.
class WeightSharedConvolutionalBoxPredictor(box_predictor.BoxPredictor):

  def __init__(self,
               is_training,
               num_classes,
               box_prediction_head,
               class_prediction_head,
               other_heads,
               conv_hyperparams_fn,
               depth,
               num_layers_before_predictor,
               kernel_size=3,
               apply_batch_norm=False,
               share_prediction_tower=False,
               use_depthwise=False):
   
    super(WeightSharedConvolutionalBoxPredictor, self).__init__(is_training,
                                                                num_classes)
    self._box_prediction_head = box_prediction_head
    self._class_prediction_head = class_prediction_head
    self._other_heads = other_heads
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._depth = depth
    self._num_layers_before_predictor = num_layers_before_predictor
    self._kernel_size = kernel_size
    self._apply_batch_norm = apply_batch_norm
    self._share_prediction_tower = share_prediction_tower
    self._use_depthwise = use_depthwise

  @property
  def num_classes(self):
    return self._num_classes

  def _insert_additional_projection_layer(self, image_feature,
                                          inserted_layer_counter,
                                          target_channel):
    if inserted_layer_counter < 0:
      return image_feature, inserted_layer_counter
    image_feature = slim.conv2d(
        image_feature,
        target_channel, [1, 1],
        stride=1,
        padding='SAME',
        activation_fn=None,
        normalizer_fn=(tf.identity if self._apply_batch_norm else None),
        scope='ProjectionLayer/conv2d_{}'.format(
            inserted_layer_counter))
    if self._apply_batch_norm:
      image_feature = slim.batch_norm(
          image_feature,
          scope='ProjectionLayer/conv2d_{}/BatchNorm'.format(
              inserted_layer_counter))
    inserted_layer_counter += 1
    return image_feature, inserted_layer_counter

  def _compute_base_tower(self, tower_name_scope, image_feature, feature_index,
                          has_different_feature_channels, target_channel,
                          inserted_layer_counter):
    net = image_feature
    for i in range(self._num_layers_before_predictor):
      if self._use_depthwise:
        conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
      else:
        conv_op = slim.conv2d
      net = conv_op(
          net,
          self._depth, [self._kernel_size, self._kernel_size],
          stride=1,
          padding='SAME',
          activation_fn=None,
          normalizer_fn=(tf.identity if self._apply_batch_norm else None),
          scope='{}/conv2d_{}'.format(tower_name_scope, i))
      if self._apply_batch_norm:
        net = slim.batch_norm(
            net,
            scope='{}/conv2d_{}/BatchNorm/feature_{}'.
            format(tower_name_scope, i, feature_index))
      net = tf.nn.relu6(net)
    return net

  def _predict_head(self, head_name, head_obj, image_feature, box_tower_feature,
                    feature_index, has_different_feature_channels,
                    target_channel, inserted_layer_counter,
                    num_predictions_per_location):
    if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
      tower_name_scope = 'ClassPredictionTower'
    else:
      raise ValueError('Unknown head')
    if self._share_prediction_tower:
      head_tower_feature = box_tower_feature
    else:
      head_tower_feature = self._compute_base_tower(
          tower_name_scope=tower_name_scope,
          image_feature=image_feature,
          feature_index=feature_index,
          has_different_feature_channels=has_different_feature_channels,
          target_channel=target_channel,
          inserted_layer_counter=inserted_layer_counter)
    return head_obj.predict(
        features=head_tower_feature,
        num_predictions_per_location=num_predictions_per_location)

  def _predict(self, image_features, num_predictions_per_location_list):
   
    if len(set(num_predictions_per_location_list)) > 1:
      raise ValueError('num predictions per location must be same for all'
                       'feature maps, found: {}'.format(
                           num_predictions_per_location_list))
    feature_channels = [
        image_feature.shape[3].value for image_feature in image_features
    ]
    has_different_feature_channels = len(set(feature_channels)) > 1
    if has_different_feature_channels:
      inserted_layer_counter = 0
      target_channel = max(set(feature_channels), key=feature_channels.count)
      tf.logging.info('Not all feature maps have the same number of '
                      'channels, found: {}, addition project layers '
                      'to bring all feature maps to uniform channels '
                      'of {}'.format(feature_channels, target_channel))
    else:
      # Place holder variables if has_different_feature_channels is False.
      target_channel = -1
      inserted_layer_counter = -1
    predictions = {
        BOX_ENCODINGS: [],
        CLASS_PREDICTIONS_WITH_BACKGROUND: [],
    }
    for head_name in self._other_heads.keys():
      predictions[head_name] = []
    for feature_index, (image_feature,
                        num_predictions_per_location) in enumerate(
                            zip(image_features,
                                num_predictions_per_location_list)):
      with tf.variable_scope('WeightSharedConvolutionalBoxPredictor',
                             reuse=tf.AUTO_REUSE):
        with slim.arg_scope(self._conv_hyperparams_fn()):
          (image_feature,
           inserted_layer_counter) = self._insert_additional_projection_layer(
               image_feature, inserted_layer_counter, target_channel)
          if self._share_prediction_tower:
            box_tower_scope = 'PredictionTower'
          else:
            box_tower_scope = 'BoxPredictionTower'
          box_tower_feature = self._compute_base_tower(
              tower_name_scope=box_tower_scope,
              image_feature=image_feature,
              feature_index=feature_index,
              has_different_feature_channels=has_different_feature_channels,
              target_channel=target_channel,
              inserted_layer_counter=inserted_layer_counter)
          box_encodings = self._box_prediction_head.predict(
              features=box_tower_feature,
              num_predictions_per_location=num_predictions_per_location)
          predictions[BOX_ENCODINGS].append(box_encodings)
          sorted_keys = sorted(self._other_heads.keys())
          sorted_keys.append(CLASS_PREDICTIONS_WITH_BACKGROUND)
          for head_name in sorted_keys:
            if head_name == CLASS_PREDICTIONS_WITH_BACKGROUND:
              head_obj = self._class_prediction_head
            else:
              head_obj = self._other_heads[head_name]
            prediction = self._predict_head(
                head_name=head_name,
                head_obj=head_obj,
                image_feature=image_feature,
                box_tower_feature=box_tower_feature,
                feature_index=feature_index,
                has_different_feature_channels=has_different_feature_channels,
                target_channel=target_channel,
                inserted_layer_counter=inserted_layer_counter,
                num_predictions_per_location=num_predictions_per_location)
            predictions[head_name].append(prediction)
    return predictions
