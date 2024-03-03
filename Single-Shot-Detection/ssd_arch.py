from abc import abstractmethod

import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils

slim = tf.contrib.slim


class SSDFeatureExtractor(object):
  """SSD Slim Feature Extractor definition."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
    
    self._is_training = is_training
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._reuse_weights = reuse_weights
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    self._override_base_feature_extractor_hyperparams = (
        override_base_feature_extractor_hyperparams)

  @property
  def is_keras_model(self):
    return False

  @abstractmethod
  def preprocess(self, resized_inputs):
   
    pass

  @abstractmethod
  def extract_features(self, preprocessed_inputs):
   
    raise NotImplementedError

  def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
    
    variables_to_restore = {}
    for variable in tf.global_variables():
      var_name = variable.op.name
      if var_name.startswith(feature_extractor_scope + '/'):
        var_name = var_name.replace(feature_extractor_scope + '/', '')
        variables_to_restore[var_name] = variable

    return variables_to_restore


class SSDKerasFeatureExtractor(tf.keras.Model):
  """SSD Feature Extractor definition."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False,
               name=None):
   
    super(SSDKerasFeatureExtractor, self).__init__(name=name)

    self._is_training = is_training
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams = conv_hyperparams
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    self._override_base_feature_extractor_hyperparams = (
        override_base_feature_extractor_hyperparams)

  @property
  def is_keras_model(self):
    return True

  @abstractmethod
  def preprocess(self, resized_inputs):
  
    raise NotImplementedError

  @abstractmethod
  def _extract_features(self, preprocessed_inputs):
    
    raise NotImplementedError

  # This overrides the keras.Model `call` method with the _extract_features
  # method.
  def call(self, inputs, **kwargs):
    return self._extract_features(inputs)

  def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
   
    variables_to_restore = {}
    for variable in tf.global_variables():
      var_name = variable.op.name
      if var_name.startswith(feature_extractor_scope + '/'):
        var_name = var_name.replace(feature_extractor_scope + '/', '')
        variables_to_restore[var_name] = variable

    return variables_to_restore


class SSDMetaArch(model.DetectionModel):
  
  def __init__(self,
               is_training,
               anchor_generator,
               box_predictor,
               box_coder,
               feature_extractor,
               encode_background_as_zeros,
               image_resizer_fn,
               non_max_suppression_fn,
               score_conversion_fn,
               classification_loss,
               localization_loss,
               classification_loss_weight,
               localization_loss_weight,
               normalize_loss_by_num_matches,
               hard_example_miner,
               target_assigner_instance,
               add_summaries=True,
               normalize_loc_loss_by_codesize=False,
               freeze_batchnorm=False,
               inplace_batchnorm_update=False,
               add_background_class=True,
               explicit_background_class=False,
               random_example_sampler=None,
               expected_loss_weights_fn=None,
               use_confidences_as_targets=False,
               implicit_example_weight=0.5,
               equalization_loss_config=None):

    super(SSDMetaArch, self).__init__(num_classes=box_predictor.num_classes)
    self._is_training = is_training
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

    self._anchor_generator = anchor_generator
    self._box_predictor = box_predictor

    self._box_coder = box_coder
    self._feature_extractor = feature_extractor
    self._add_background_class = add_background_class
    self._explicit_background_class = explicit_background_class

    if add_background_class and explicit_background_class:
      raise ValueError("Cannot have both 'add_background_class' and"
                       " 'explicit_background_class' true.")

    # Needed for fine-tuning from classification checkpoints whose
    # variables do not have the feature extractor scope.
    if self._feature_extractor.is_keras_model:
      # Keras feature extractors will have a name they implicitly use to scope.
      # So, all contained variables are prefixed by this name.
      # To load from classification checkpoints, need to filter out this name.
      self._extract_features_scope = feature_extractor.name
    else:
      # Slim feature extractors get an explicit naming scope
      self._extract_features_scope = 'FeatureExtractor'

    if encode_background_as_zeros:
      background_class = [0]
    else:
      background_class = [1]

    if self._add_background_class:
      num_foreground_classes = self.num_classes
    else:
      num_foreground_classes = self.num_classes - 1

    self._unmatched_class_label = tf.constant(
        background_class + num_foreground_classes * [0], tf.float32)

    self._target_assigner = target_assigner_instance

    self._classification_loss = classification_loss
    self._localization_loss = localization_loss
    self._classification_loss_weight = classification_loss_weight
    self._localization_loss_weight = localization_loss_weight
    self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
    self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize
    self._hard_example_miner = hard_example_miner
    self._random_example_sampler = random_example_sampler
    self._parallel_iterations = 16

    self._image_resizer_fn = image_resizer_fn
    self._non_max_suppression_fn = non_max_suppression_fn
    self._score_conversion_fn = score_conversion_fn

    self._anchors = None
    self._add_summaries = add_summaries
    self._batched_prediction_tensor_names = []
    self._expected_loss_weights_fn = expected_loss_weights_fn
    self._use_confidences_as_targets = use_confidences_as_targets
    self._implicit_example_weight = implicit_example_weight

    self._equalization_loss_config = equalization_loss_config

  @property
  def anchors(self):
    if not self._anchors:
      raise RuntimeError('anchors have not been constructed yet!')
    if not isinstance(self._anchors, box_list.BoxList):
      raise RuntimeError('anchors should be a BoxList object, but is not.')
    return self._anchors

  @property
  def batched_prediction_tensor_names(self):
    if not self._batched_prediction_tensor_names:
      raise RuntimeError('Must call predict() method to get batched prediction '
                         'tensor names.')
    return self._batched_prediction_tensor_names

  def preprocess(self, inputs):
   
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocessor'):
      # TODO(jonathanhuang): revisit whether to always use batch size as
      # the number of parallel iterations vs allow for dynamic batching.
      outputs = shape_utils.static_or_dynamic_map_fn(
          self._image_resizer_fn,
          elems=inputs,
          dtype=[tf.float32, tf.int32])
      resized_inputs = outputs[0]
      true_image_shapes = outputs[1]

      return (self._feature_extractor.preprocess(resized_inputs),
              true_image_shapes)

  def _compute_clip_window(self, preprocessed_images, true_image_shapes):
   
    if true_image_shapes is None:
      return tf.constant([0, 0, 1, 1], dtype=tf.float32)

    resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_images)
    true_heights, true_widths, _ = tf.unstack(
        tf.to_float(true_image_shapes), axis=1)
    padded_height = tf.to_float(resized_inputs_shape[1])
    padded_width = tf.to_float(resized_inputs_shape[2])
    return tf.stack(
        [
            tf.zeros_like(true_heights),
            tf.zeros_like(true_widths), true_heights / padded_height,
            true_widths / padded_width
        ],
        axis=1)

  def predict(self, preprocessed_inputs, true_image_shapes):
   
    batchnorm_updates_collections = (None if self._inplace_batchnorm_update
                                     else tf.GraphKeys.UPDATE_OPS)
    if self._feature_extractor.is_keras_model:
      feature_maps = self._feature_extractor(preprocessed_inputs)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        with tf.variable_scope(None, self._extract_features_scope,
                               [preprocessed_inputs]):
          feature_maps = self._feature_extractor.extract_features(
              preprocessed_inputs)

    feature_map_spatial_dims = self._get_feature_map_spatial_dims(
        feature_maps)
    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)
    self._anchors = box_list_ops.concatenate(
        self._anchor_generator.generate(
            feature_map_spatial_dims,
            im_height=image_shape[1],
            im_width=image_shape[2]))
    if self._box_predictor.is_keras_model:
      predictor_results_dict = self._box_predictor(feature_maps)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        predictor_results_dict = self._box_predictor.predict(
            feature_maps, self._anchor_generator.num_anchors_per_location())
    predictions_dict = {
        'preprocessed_inputs': preprocessed_inputs,
        'feature_maps': feature_maps,
        'anchors': self._anchors.get()
    }
    for prediction_key, prediction_list in iter(predictor_results_dict.items()):
      prediction = tf.concat(prediction_list, axis=1)
      if (prediction_key == 'box_encodings' and prediction.shape.ndims == 4 and
          prediction.shape[2] == 1):
        prediction = tf.squeeze(prediction, axis=2)
      predictions_dict[prediction_key] = prediction
    self._batched_prediction_tensor_names = [x for x in predictions_dict
                                             if x != 'anchors']
    return predictions_dict

  def _get_feature_map_spatial_dims(self, feature_maps):
   
    feature_map_shapes = [
        shape_utils.combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]

  def postprocess(self, prediction_dict, true_image_shapes):
   
    if ('box_encodings' not in prediction_dict or
        'class_predictions_with_background' not in prediction_dict):
      raise ValueError('prediction_dict does not contain expected entries.')
    with tf.name_scope('Postprocessor'):
      preprocessed_images = prediction_dict['preprocessed_inputs']
      box_encodings = prediction_dict['box_encodings']
      box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
      class_predictions = prediction_dict['class_predictions_with_background']
      detection_boxes, detection_keypoints = self._batch_decode(box_encodings)
      detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
      detection_boxes = tf.expand_dims(detection_boxes, axis=2)

      detection_scores = self._score_conversion_fn(class_predictions)
      detection_scores = tf.identity(detection_scores, 'raw_box_scores')
      if self._add_background_class or self._explicit_background_class:
        detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])
      additional_fields = None

      batch_size = (
          shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0])

      if 'feature_maps' in prediction_dict:
        feature_map_list = []
        for feature_map in prediction_dict['feature_maps']:
          feature_map_list.append(tf.reshape(feature_map, [batch_size, -1]))
        box_features = tf.concat(feature_map_list, 1)
        box_features = tf.identity(box_features, 'raw_box_features')

      if detection_keypoints is not None:
        additional_fields = {
            fields.BoxListFields.keypoints: detection_keypoints}
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields, num_detections) = self._non_max_suppression_fn(
           detection_boxes,
           detection_scores,
           clip_window=self._compute_clip_window(preprocessed_images,
                                                 true_image_shapes),
           additional_fields=additional_fields,
           masks=prediction_dict.get('mask_predictions'))
      detection_dict = {
          fields.DetectionResultFields.detection_boxes: nmsed_boxes,
          fields.DetectionResultFields.detection_scores: nmsed_scores,
          fields.DetectionResultFields.detection_classes: nmsed_classes,
          fields.DetectionResultFields.num_detections:
              tf.to_float(num_detections)
      }
      if (nmsed_additional_fields is not None and
          fields.BoxListFields.keypoints in nmsed_additional_fields):
        detection_dict[fields.DetectionResultFields.detection_keypoints] = (
            nmsed_additional_fields[fields.BoxListFields.keypoints])
      if nmsed_masks is not None:
        detection_dict[
            fields.DetectionResultFields.detection_masks] = nmsed_masks
      return detection_dict

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      keypoints = None
      if self.groundtruth_has_field(fields.BoxListFields.keypoints):
        keypoints = self.groundtruth_lists(fields.BoxListFields.keypoints)
      weights = None
      if self.groundtruth_has_field(fields.BoxListFields.weights):
        weights = self.groundtruth_lists(fields.BoxListFields.weights)
      confidences = None
      if self.groundtruth_has_field(fields.BoxListFields.confidences):
        confidences = self.groundtruth_lists(fields.BoxListFields.confidences)
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, match_list) = self._assign_targets(
           self.groundtruth_lists(fields.BoxListFields.boxes),
           self.groundtruth_lists(fields.BoxListFields.classes),
           keypoints, weights, confidences)
      if self._add_summaries:
        self._summarize_target_assignment(
            self.groundtruth_lists(fields.BoxListFields.boxes), match_list)

      if self._random_example_sampler:
        batch_cls_per_anchor_weights = tf.reduce_mean(
            batch_cls_weights, axis=-1)
        batch_sampled_indicator = tf.to_float(
            shape_utils.static_or_dynamic_map_fn(
                self._minibatch_subsample_fn,
                [batch_cls_targets, batch_cls_per_anchor_weights],
                dtype=tf.bool,
                parallel_iterations=self._parallel_iterations,
                back_prop=True))
        batch_reg_weights = tf.multiply(batch_sampled_indicator,
                                        batch_reg_weights)
        batch_cls_weights = tf.multiply(
            tf.expand_dims(batch_sampled_indicator, -1),
            batch_cls_weights)

      losses_mask = None
      if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
        losses_mask = tf.stack(self.groundtruth_lists(
            fields.InputDataFields.is_annotated))
      location_losses = self._localization_loss(
          prediction_dict['box_encodings'],
          batch_reg_targets,
          ignore_nan_targets=True,
          weights=batch_reg_weights,
          losses_mask=losses_mask)

      cls_losses = self._classification_loss(
          prediction_dict['class_predictions_with_background'],
          batch_cls_targets,
          weights=batch_cls_weights,
          losses_mask=losses_mask)

      if self._expected_loss_weights_fn:
        # Need to compute losses for assigned targets against the
        # unmatched_class_label as well as their assigned targets.
        # simplest thing (but wasteful) is just to calculate all losses
        # twice
        batch_size, num_anchors, num_classes = batch_cls_targets.get_shape()
        unmatched_targets = tf.ones([batch_size, num_anchors, 1
                                    ]) * self._unmatched_class_label

        unmatched_cls_losses = self._classification_loss(
            prediction_dict['class_predictions_with_background'],
            unmatched_targets,
            weights=batch_cls_weights,
            losses_mask=losses_mask)

        if cls_losses.get_shape().ndims == 3:
          batch_size, num_anchors, num_classes = cls_losses.get_shape()
          cls_losses = tf.reshape(cls_losses, [batch_size, -1])
          unmatched_cls_losses = tf.reshape(unmatched_cls_losses,
                                            [batch_size, -1])
          batch_cls_targets = tf.reshape(
              batch_cls_targets, [batch_size, num_anchors * num_classes, -1])
          batch_cls_targets = tf.concat(
              [1 - batch_cls_targets, batch_cls_targets], axis=-1)

          location_losses = tf.tile(location_losses, [1, num_classes])

        foreground_weights, background_weights = (
            self._expected_loss_weights_fn(batch_cls_targets))

        cls_losses = (
            foreground_weights * cls_losses +
            background_weights * unmatched_cls_losses)

        location_losses *= foreground_weights

        classification_loss = tf.reduce_sum(cls_losses)
        localization_loss = tf.reduce_sum(location_losses)
      elif self._hard_example_miner:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        (localization_loss, classification_loss) = self._apply_hard_mining(
            location_losses, cls_losses, prediction_dict, match_list)
        if self._add_summaries:
          self._hard_example_miner.summarize()
      else:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        localization_loss = tf.reduce_sum(location_losses)
        classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if self._normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.to_float(tf.reduce_sum(batch_reg_weights)),
                                1.0)

      localization_loss_normalizer = normalizer
      if self._normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= self._box_coder.code_size
      localization_loss = tf.multiply((self._localization_loss_weight /
                                       localization_loss_normalizer),
                                      localization_loss,
                                      name='localization_loss')
      classification_loss = tf.multiply((self._classification_loss_weight /
                                         normalizer), classification_loss,
                                        name='classification_loss')

      loss_dict = {
          str(localization_loss.op.name): localization_loss,
          str(classification_loss.op.name): classification_loss
      }


    return loss_dict

  def _minibatch_subsample_fn(self, inputs):
   
    cls_targets, cls_weights = inputs
    if self._add_background_class:
      # Set background_class bits to 0 so that the positives_indicator
      # computation would not consider background class.
      background_class = tf.zeros_like(tf.slice(cls_targets, [0, 0], [-1, 1]))
      regular_class = tf.slice(cls_targets, [0, 1], [-1, -1])
      cls_targets = tf.concat([background_class, regular_class], 1)
    positives_indicator = tf.reduce_sum(cls_targets, axis=1)
    return self._random_example_sampler.subsample(
        tf.cast(cls_weights, tf.bool),
        batch_size=None,
        labels=tf.cast(positives_indicator, tf.bool))

  def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
    positive_indices = tf.where(tf.greater(class_ids, 0))
    positive_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, positive_indices), axis=1)
    visualization_utils.add_cdf_image_summary(positive_anchor_cls_loss,
                                              'PositiveAnchorLossCDF')
    negative_indices = tf.where(tf.equal(class_ids, 0))
    negative_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, negative_indices), axis=1)
    visualization_utils.add_cdf_image_summary(negative_anchor_cls_loss,
                                              'NegativeAnchorLossCDF')

  def _assign_targets(self,
                      groundtruth_boxes_list,
                      groundtruth_classes_list,
                      groundtruth_keypoints_list=None,
                      groundtruth_weights_list=None,
                      groundtruth_confidences_list=None):
    
    groundtruth_boxlists = [
        box_list.BoxList(boxes) for boxes in groundtruth_boxes_list
    ]
    train_using_confidences = (self._is_training and
                               self._use_confidences_as_targets)
    if self._add_background_class:
      groundtruth_classes_with_background_list = [
          tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
          for one_hot_encoding in groundtruth_classes_list
      ]
      if train_using_confidences:
        groundtruth_confidences_with_background_list = [
            tf.pad(groundtruth_confidences, [[0, 0], [1, 0]], mode='CONSTANT')
            for groundtruth_confidences in groundtruth_confidences_list
        ]
    else:
      groundtruth_classes_with_background_list = groundtruth_classes_list

    if groundtruth_keypoints_list is not None:
      for boxlist, keypoints in zip(
          groundtruth_boxlists, groundtruth_keypoints_list):
        boxlist.add_field(fields.BoxListFields.keypoints, keypoints)
    if train_using_confidences:
      return target_assigner.batch_assign_confidences(
          self._target_assigner,
          self.anchors,
          groundtruth_boxlists,
          groundtruth_confidences_with_background_list,
          groundtruth_weights_list,
          self._unmatched_class_label,
          self._add_background_class,
          self._implicit_example_weight)
    else:
      return target_assigner.batch_assign_targets(
          self._target_assigner,
          self.anchors,
          groundtruth_boxlists,
          groundtruth_classes_with_background_list,
          self._unmatched_class_label,
          groundtruth_weights_list)

  def _summarize_target_assignment(self, groundtruth_boxes_list, match_list):
   
    num_boxes_per_image = tf.stack(
        [tf.shape(x)[0] for x in groundtruth_boxes_list])
    pos_anchors_per_image = tf.stack(
        [match.num_matched_columns() for match in match_list])
    neg_anchors_per_image = tf.stack(
        [match.num_unmatched_columns() for match in match_list])
    ignored_anchors_per_image = tf.stack(
        [match.num_ignored_columns() for match in match_list])
    tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                      tf.reduce_mean(tf.to_float(num_boxes_per_image)),
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                      tf.reduce_mean(tf.to_float(pos_anchors_per_image)),
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                      tf.reduce_mean(tf.to_float(neg_anchors_per_image)),
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumIgnoredAnchorsPerImage',
                      tf.reduce_mean(tf.to_float(ignored_anchors_per_image)),
                      family='TargetAssignment')

  def _apply_hard_mining(self, location_losses, cls_losses, prediction_dict,
                         match_list):
    
    class_predictions = tf.slice(
        prediction_dict['class_predictions_with_background'], [0, 0,
                                                               1], [-1, -1, -1])

    decoded_boxes, _ = self._batch_decode(prediction_dict['box_encodings'])
    decoded_box_tensors_list = tf.unstack(decoded_boxes)
    class_prediction_list = tf.unstack(class_predictions)
    decoded_boxlist_list = []
    for box_location, box_score in zip(decoded_box_tensors_list,
                                       class_prediction_list):
      decoded_boxlist = box_list.BoxList(box_location)
      decoded_boxlist.add_field('scores', box_score)
      decoded_boxlist_list.append(decoded_boxlist)
    return self._hard_example_miner(
        location_losses=location_losses,
        cls_losses=cls_losses,
        decoded_boxlist_list=decoded_boxlist_list,
        match_list=match_list)

  def _batch_decode(self, box_encodings):
   
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(
        tf.expand_dims(self.anchors.get(), 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))
    decoded_boxes = self._box_coder.decode(
        tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
        tiled_anchors_boxlist)
    decoded_keypoints = None
    if decoded_boxes.has_field(fields.BoxListFields.keypoints):
      decoded_keypoints = decoded_boxes.get_field(
          fields.BoxListFields.keypoints)
      num_keypoints = decoded_keypoints.get_shape()[1]
      decoded_keypoints = tf.reshape(
          decoded_keypoints,
          tf.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
    decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes, decoded_keypoints

  def regularization_losses(self):
    """Returns a list of regularization losses for this model.

    Returns a list of regularization losses for this model that the estimator
    needs to use during training/optimization.

    Returns:
      A list of regularization loss tensors.
    """
    losses = []
    slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Copy the slim losses to avoid modifying the collection
    if slim_losses:
      losses.extend(slim_losses)
    if self._box_predictor.is_keras_model:
      losses.extend(self._box_predictor.losses)
    if self._feature_extractor.is_keras_model:
      losses.extend(self._feature_extractor.losses)
    return losses

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False):
    

    if fine_tune_checkpoint_type not in ['detection', 'classification']:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))

    if fine_tune_checkpoint_type == 'classification':
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          self._extract_features_scope)

    if fine_tune_checkpoint_type == 'detection':
      variables_to_restore = {}
      for variable in tf.global_variables():
        var_name = variable.op.name
        if load_all_detection_checkpoint_vars:
          variables_to_restore[var_name] = variable
        else:
          if var_name.startswith(self._extract_features_scope):
            variables_to_restore[var_name] = variable

    return variables_to_restore

  def updates(self):
   
    update_ops = []
    slim_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Copy the slim ops to avoid modifying the collection
    if slim_update_ops:
      update_ops.extend(slim_update_ops)
    if self._box_predictor.is_keras_model:
      update_ops.extend(self._box_predictor.get_updates_for(None))
      update_ops.extend(self._box_predictor.get_updates_for(
          self._box_predictor.inputs))
    if self._feature_extractor.is_keras_model:
      update_ops.extend(self._feature_extractor.get_updates_for(None))
      update_ops.extend(self._feature_extractor.get_updates_for(
          self._feature_extractor.inputs))
    return update_ops
