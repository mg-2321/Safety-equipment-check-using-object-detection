import tensorflow as tf

from object_detection.core import box_coder
from object_detection.core import box_list

EPSILON = 1e-8


class SquareBoxCoder(box_coder.BoxCoder):
  """Encodes a 3-scalar representation of a square box."""

  def __init__(self, scale_factors=None):
    """Constructor for SquareBoxCoder.

    Args:
      scale_factors: List of 3 positive scalars to scale ty, tx, and tl.
        If set to None, does not perform scaling. For faster RCNN,
        the open-source implementation recommends using [10.0, 10.0, 5.0].

    Raises:
      ValueError: If scale_factors is not length 3 or contains values less than
        or equal to 0.
    """
    if scale_factors:
      if len(scale_factors) != 3:
        raise ValueError('The argument scale_factors must be a list of length '
                         '3.')
      if any(scalar <= 0 for scalar in scale_factors):
        raise ValueError('The values in scale_factors must all be greater '
                         'than 0.')
    self._scale_factors = scale_factors

  @property
  def code_size(self):
    return 3

  def _encode(self, boxes, anchors):
    """Encodes a box collection with respect to an anchor collection.

    Args:
      boxes: BoxList holding N boxes to be encoded.
      anchors: BoxList of anchors.

    Returns:
      a tensor representing N anchor-encoded boxes of the format
      [ty, tx, tl].
    """
    # Convert anchors to the center coordinate representation.
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    la = tf.sqrt(ha * wa)
    ycenter, xcenter, h, w = boxes.get_center_coordinates_and_sizes()
    l = tf.sqrt(h * w)
    # Avoid NaN in division and log below.
    la += EPSILON
    l += EPSILON

    tx = (xcenter - xcenter_a) / la
    ty = (ycenter - ycenter_a) / la
    tl = tf.log(l / la)
    # Scales location targets for joint training.
    if self._scale_factors:
      ty *= self._scale_factors[0]
      tx *= self._scale_factors[1]
      tl *= self._scale_factors[2]
    return tf.transpose(tf.stack([ty, tx, tl]))

  def _decode(self, rel_codes, anchors):
    """Decodes relative codes to boxes.

    Args:
      rel_codes: a tensor representing N anchor-encoded boxes.
      anchors: BoxList of anchors.

    Returns:
      boxes: BoxList holding N bounding boxes.
    """
    ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
    la = tf.sqrt(ha * wa)

    ty, tx, tl = tf.unstack(tf.transpose(rel_codes))
    if self._scale_factors:
      ty /= self._scale_factors[0]
      tx /= self._scale_factors[1]
      tl /= self._scale_factors[2]
    l = tf.exp(tl) * la
    ycenter = ty * la + ycenter_a
    xcenter = tx * la + xcenter_a
    ymin = ycenter - l / 2.
    xmin = xcenter - l / 2.
    ymax = ycenter + l / 2.
    xmax = xcenter + l / 2.
    return box_list.BoxList(tf.transpose(tf.stack([ymin, xmin, ymax, xmax])))
