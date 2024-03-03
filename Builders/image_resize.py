import functools
import tensorflow as tf

from object_detection.core import preprocessor
from object_detection.protos import image_resizer_pb2


def _tf_resize_method(resize_method):
  
  dict_method = {
      image_resizer_pb2.BILINEAR:
          tf.image.ResizeMethod.BILINEAR,
      image_resizer_pb2.NEAREST_NEIGHBOR:
          tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      image_resizer_pb2.BICUBIC:
          tf.image.ResizeMethod.BICUBIC,
      image_resizer_pb2.AREA:
          tf.image.ResizeMethod.AREA
  }
  if resize_method in dict_method:
    return dict_method[resize_method]
  else:
    raise ValueError('Unknown resize_method')


def build(image_resizer_config):
  
  if not isinstance(image_resizer_config, image_resizer_pb2.ImageResizer):
    raise ValueError('image_resizer_config not of type '
                     'image_resizer_pb2.ImageResizer.')

  image_resizer_oneof = image_resizer_config.WhichOneof('image_resizer_oneof')
  if image_resizer_oneof == 'keep_aspect_ratio_resizer':
    keep_aspect_ratio_config = image_resizer_config.keep_aspect_ratio_resizer
    if not (keep_aspect_ratio_config.min_dimension <=
            keep_aspect_ratio_config.max_dimension):
      raise ValueError('min_dimension > max_dimension')
    method = _tf_resize_method(keep_aspect_ratio_config.resize_method)
    per_channel_pad_value = (0, 0, 0)
    if keep_aspect_ratio_config.per_channel_pad_value:
      per_channel_pad_value = tuple(keep_aspect_ratio_config.
                                    per_channel_pad_value)
    image_resizer_fn = functools.partial(
        preprocessor.resize_to_range,
        min_dimension=keep_aspect_ratio_config.min_dimension,
        max_dimension=keep_aspect_ratio_config.max_dimension,
        method=method,
        pad_to_max_dimension=keep_aspect_ratio_config.pad_to_max_dimension,
        per_channel_pad_value=per_channel_pad_value)
    if not keep_aspect_ratio_config.convert_to_grayscale:
      return image_resizer_fn
  elif image_resizer_oneof == 'fixed_shape_resizer':
    fixed_shape_resizer_config = image_resizer_config.fixed_shape_resizer
    method = _tf_resize_method(fixed_shape_resizer_config.resize_method)
    image_resizer_fn = functools.partial(
        preprocessor.resize_image,
        new_height=fixed_shape_resizer_config.height,
        new_width=fixed_shape_resizer_config.width,
        method=method)
    if not fixed_shape_resizer_config.convert_to_grayscale:
      return image_resizer_fn
  else:
    raise ValueError(
        'Invalid image resizer option: \'%s\'.' % image_resizer_oneof)

  def grayscale_image_resizer(image, masks=None):
    """Convert to grayscale before applying image_resizer_fn.

   
    # image_resizer_fn returns [resized_image, resized_image_shape] if
    # mask==None, otherwise it returns
    # [resized_image, resized_mask, resized_image_shape]. In either case, we
    # only deal with first and last element of the returned list.
    retval = image_resizer_fn(image, masks)
    resized_image = retval[0]
    resized_image_shape = retval[-1]
    retval[0] = preprocessor.rgb_to_gray(resized_image)
    retval[-1] = tf.concat([resized_image_shape[:-1], [1]], 0)
    return retval

  return functools.partial(grayscale_image_resizer)
