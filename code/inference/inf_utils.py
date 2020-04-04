# coding=utf-8
"""Utils."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
import os
import sys
import cv2
import tensorflow as tf
import numpy as np
import json
import operator
import itertools
import collections
import math

from PIL import Image

# Dependencies - 1: Object Detection & Tracking
# Assuming object detection repo is downloaded under the inferencing folder
# $ git clone https://github.com/JunweiLiang/Object_Detection_Tracking
# $ cd  Object_Detection_Tracking
# $ git checkout 2b9622456218a114a66d71972c002a4f9897b77f
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Object_Detection_Tracking"))
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from application_util import preprocessing
from deep_sort.utils import create_obj_infos
from deep_sort.utils import linear_inter_bbox
from deep_sort.utils import filter_short_objs
from class_ids import targetClass2id_new_nopo
from class_ids import coco_obj_to_actev_obj
from nn import resnet_fpn_backbone
from nn import fpn_model

# The next-prediction code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pred_models import Model as PredictionModel
from pred_utils import activity2id


def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


def find_past_trajs(traj_list, start_frame_idx, num_past_steps):
  # assume traj_list is sorted by start_frame_idx and is consecutive
  this_traj_idx = [i for i, (fidx, seqid, idx) in enumerate(traj_list)
                  if fidx == start_frame_idx][0]
  # note this idxs is the idxs for traj_list,
  # we need the idx within traj_list
  prev_idxs = []

  for i in range(1, num_past_steps + 1):
    idx = this_traj_idx - i
    if idx < 0:
      break
    prev_idxs.append((idx, - i))
  return [(traj_list[idx][2], rel) for idx, rel in prev_idxs]


def get_video_meta(vcap):
  """Given the cv2 opened video, get video metadata."""
  if cv2.__version__.split(".")[0] != "2":
    frame_width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = vcap.get(cv2.CAP_PROP_FPS)
    frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
  else:
    frame_width = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_height = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    fps = vcap.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_count = vcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
  return {
      "frame_height": frame_height,
      "frame_width": frame_width,
      "fps": fps,
      "frame_count": frame_count}


def resize_img(img_data, max_size, shorter_edge_size, force=False):
  """Resize img_data of [H, W, 3] to a new size."""
  new_img_w, new_img_h = get_new_hw(img_data.shape[0], img_data.shape[1],
                                    max_size, shorter_edge_size)

  if force:
    new_img_data = cv2.resize(img_data, (max_size, shorter_edge_size),
                              interpolation=cv2.INTER_LINEAR)
  else:
    new_img_data = cv2.resize(img_data, (new_img_w, new_img_h),
                              interpolation=cv2.INTER_LINEAR)

  return new_img_data


def get_new_hw(h, w, max_size, shorter_edge_size):
  """Get the new img size with the same ratio."""

  scale = shorter_edge_size * 1.0 / min(h, w)
  if h < w:
    newh, neww = shorter_edge_size, scale * w
  else:
    newh, neww = scale * h, shorter_edge_size
  if max(newh, neww) > max_size:
    scale = max_size * 1.0 / max(newh, neww)
    newh = newh * scale
    neww = neww * scale
  neww = int(neww + 0.5)
  newh = int(newh + 0.5)
  return neww, newh


def get_scene_seg_model(model_path, gpuid):
  with tf.device("/gpu:%s" % gpuid):
    model = SceneSeg(model_path, gpuid)
  return model


class SceneSeg(object):
  """Scene semantic segmentation class."""

  def __init__(self, model_path, gpuid):
    self.graph = tf.get_default_graph()

    with tf.gfile.GFile(model_path, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    self.var_prefix = "scene_seg_model_%s" % gpuid
    tf.import_graph_def(
        graph_def,
        name=self.var_prefix,
        return_elements=None
    )

    # input place holders
    self.input_tensor = self.graph.get_tensor_by_name(
        "%s/ImageTensor:0" % self.var_prefix)
    self.output_tensor = self.graph.get_tensor_by_name(
        "%s/SemanticPredictions:0" % self.var_prefix)

    self.input_size = 513  # this depends on the deeplabv3 model

  def get_feed_dict_forward(self, img_file):
    """Get feed dict input."""
    feed_dict = {}

    ori_img = Image.open(img_file)

    w, h = ori_img.size
    resize_r = 1.0 * self.input_size / max(w, h)
    target_size = (int(resize_r * w), int(resize_r * h))
    resized_imgdata = ori_img.convert("RGB").resize(target_size,
                                                    Image.ANTIALIAS)

    feed_dict[self.input_tensor] = [np.asarray(resized_imgdata)]

    return feed_dict


def resize_seg_map(seg, down_rate):
  """Given seg tensor, resize."""
  img_ = Image.fromarray(seg.astype(dtype=np.uint8))
  w_, h_ = img_.size
  neww, newh = int(w_ / down_rate), int(h_ / down_rate)

  newimg = img_.resize((neww, newh))  # neareast neighbor

  newdata = np.array(newimg)
  return newdata


class DeepSortTracker(object):
  """A high level wrapper for deep sort tracking using object detection feat."""

  def __init__(self, track_obj="Person", metric="cosine", max_cosine_dist=0.5,
               min_detection_height=0, min_confidence=0.85,
               nms_max_overlap=0.85, nn_budget=5, frame_gap=1,
               is_coco_class=False,
               is_partial_model=False):

    if is_partial_model:
      assert is_coco_class
      self.partial_classes = [classname for classname in coco_obj_to_actev_obj]
      partial_classes = ["BG"] + self.partial_classes
      partial_obj_class2id = {classname: i
                              for i, classname in enumerate(partial_classes)}
      self.partial_obj_id2class = {
          partial_obj_class2id[o]: o for o in partial_obj_class2id}

    self.track_obj = track_obj
    self.min_confidence = min_confidence
    self.min_detection_height = min_detection_height
    self.nms_max_overlap = nms_max_overlap
    self.frame_gap = frame_gap

    metric = nn_matching.NearestNeighborDistanceMetric(metric, max_cosine_dist,
                                                       nn_budget)
    # initialize result storage.
    self.tracker = Tracker(metric)
    self.tracking_results = []
    self.tmp_tracking_dict = {}

    self.is_coco_class = is_coco_class

  def track(self, boxes, labels, probs, box_feats, frame_num):
    """Given new object detection output, update the tracking."""
    obj_id_to_class_ = obj_id_to_class
    if self.is_coco_class:
      obj_id_to_class_ = coco_obj_id_to_class
      if self.partial_classes:
        obj_id_to_class_ = self.partial_obj_id2class
    detections = create_obj_infos(frame_num, boxes, probs, labels, box_feats,
                                  obj_id_to_class_, [self.track_obj],
                                  self.min_confidence,
                                  self.min_detection_height, 1.0,
                                  is_coco_model=self.is_coco_class,
                                  coco_to_actev_mapping=coco_obj_to_actev_obj)
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, self.nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # update tracker
    self.tracker.predict()
    self.tracker.update(detections)

    # store result
    for track in self.tracker.tracks:
      if not track.is_confirmed() or track.time_since_update > 1:

        if not track.is_confirmed and track.time_since_update == 0:
          bbox = track.to_tlwh()
          dp = [frame_num, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
          if track.track_id not in self.tmp_tracking_dict:
            self.tmp_tracking_dict[track.track_id] = [dp]
          else:
            self.tmp_tracking_dict[track.track_id].append(dp)
        continue

      bbox = track.to_tlwh()
      if track.track_id in self.tmp_tracking_dict:
        pred_list = self.tmp_tracking_dict[track.track_id]
        for dp in pred_list:
          self.tracking_results.append(dp)
        self.tmp_tracking_dict[track.track_id].pop(track.track_id, None)
      self.tracking_results.append([frame_num, track.track_id, bbox[0], bbox[1],
                                    bbox[2], bbox[3]])

  def finalize(self):
    """Refine the results and return MOT format output."""
    tracking_results = sorted(self.tracking_results, key=lambda x: (x[0], x[1]))
    tracking_results = np.asarray(tracking_results)
    tracking_results = linear_inter_bbox(tracking_results, self.frame_gap)
    tracking_results = filter_short_objs(tracking_results)

    # [frameIdx, track_id, left, top, width, height]
    return tracking_results.tolist()


def get_coco_list(boxes, labels, probs, is_coco_class=False):
  """Given the object model output tensors, get a list in coco format."""
  pred = []

  for box, prob, label in zip(boxes, probs, labels):
    box[2] -= box[0]
    box[3] -= box[1]  # produce x,y,w,h output

    cat_id = int(label)
    if is_coco_class:
      cat_name = coco_obj_id_to_class[cat_id]
    else:
      cat_name = obj_id_to_class[cat_id]

    # encode mask
    rle = None

    res = {
        "category_id": cat_id,
        "cat_name": cat_name,  # [0-80]
        "score": float(round(prob, 7)),
        "bbox": [float(round(x, 2)) for x in box],
        "segmentation": rle,
    }

    pred.append(res)
  return pred


def get_traj_point(box):
  """Given [x, y, w, h] person box, get person traj."""
  x, y, w, h = box
  return [(x + w + x) / 2.0, y + h]


def convert_box(box):
  """[x, y, w, h] to x1, y1, x2, y2."""
  x, y, w, h = box
  return [x, y, x + w, y + h]


def clip_box(box, max_width, max_height):
  """clipping a [x,y,w,h] boxes."""
  x, y, w, h = box

  if x + w > max_width:
    w = max_width - x
    w = 0.0 if w < 0 else w
  if y + h > max_height:
    h = max_height - y
    h = 0.0 if h < 0 else h

  if x > max_width:
    x = max_width
  if y > max_height:
    y = max_height

  return [x, y, w, h]


def load_obj_boxes(json_file, topk, object2id):
  """Load object from COCO json."""
  with open(json_file, "r") as f:
    data = json.load(f)
  newdata = []
  for one in data:
    if one["cat_name"] in object2id:
      one["bbox"] = convert_box(one["bbox"])
      newdata.append(one)

  newdata.sort(key=operator.itemgetter("score"), reverse=True)
  return newdata[:topk]


def get_nearest(frame_idxs, frame_idx):
  """Since we don"t run scene seg on every frame,we want to find the nearest."""
  frame_idxs = np.array(frame_idxs)
  cloests_i = (np.abs(frame_idxs - frame_idx)).argmin()
  return frame_idxs[cloests_i]


# ------------------------ for extracting person appearance feature


def get_person_appearance_model(model_config, sess, gpuid=0):
  with tf.device("/gpu:%s" % gpuid):
    model = ModelFPN(model_config)
  load_model_weights(model_config.model_path, sess)
  return model


# pylint: disable=g-line-too-long
def roi_align(featuremap, boxes, output_shape_h, output_shape_w):
  """Modified roi_align to allow for non-rectangle output shape. Origin: https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_box.py ."""
  boxes = tf.stop_gradient(boxes)
  box_ind = tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32)

  image_shape = tf.shape(featuremap)[2:]
  crop_shape = [output_shape_h * 2, output_shape_w * 2]
  x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

  spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
  spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

  nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
  ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

  nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(
      image_shape[1] - 1)
  nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(
      image_shape[0] - 1)

  boxes = tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

  featuremap = tf.transpose(featuremap, [0, 2, 3, 1])
  ret = tf.image.crop_and_resize(
      featuremap, boxes, box_ind,
      crop_size=[crop_shape[0], crop_shape[1]])
  ret = tf.transpose(ret, [0, 3, 1, 2])

  ret = tf.nn.avg_pool(ret, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2],
                       padding="SAME", data_format="NCHW")
  return ret


def load_model_weights(model_path, sess, top_scope=None):
  """Load model weights into tf Graph."""

  tf.global_variables_initializer().run()
  allvars = tf.global_variables()
  allvars = [var for var in allvars if "global_step" not in var.name]
  restore_vars = allvars
  opts = ["Adam", "beta1_power", "beta2_power", "Adam_1", "Adadelta_1",
          "Adadelta", "Momentum"]
  restore_vars = [var for var in restore_vars
                  if var.name.split(":")[0].split("/")[-1] not in opts]

  if top_scope is not None:
    restore_vars = [var for var in restore_vars
                    if var.name.split(":")[0].split("/")[0] == top_scope]
  saver = tf.train.Saver(restore_vars, max_to_keep=5)

  load_from = model_path

  ckpt = tf.train.get_checkpoint_state(load_from)
  if ckpt and ckpt.model_checkpoint_path:
    loadpath = ckpt.model_checkpoint_path
    saver.restore(sess, loadpath)
  else:
    raise Exception("Model not exists")


# pylint: disable=invalid-name
class ModelFPN(object):
  """FPN backbone model for extracting features given boxes."""

  def __init__(self, config):
    self.config = config
    H = self.imgh = config.imgh
    W = self.imgw = config.imgw
    # image input, one image per person box
    # [H,W,3]
    self.imgs = tf.placeholder("float32", [1, H, W, 3], name="img")
    # [4]
    self.boxes = tf.placeholder("float32", [None, 4], name="person_box")

    # the model graph
    with tf.name_scope("image_prepro"):
      images = self.imgs
      mean = [0.485, 0.456, 0.406]
      std = [0.229, 0.224, 0.225]
      # cv2 load image is bgr
      mean = mean[::-1]
      std = std[::-1]
      image_mean = tf.constant(mean, dtype=tf.float32)
      image_std = tf.constant(std, dtype=tf.float32)

      images = images*(1.0/255)
      images = (images - image_mean) / image_std
      images = tf.transpose(images, [0, 3, 1, 2])

    with tf.name_scope("fpn_backbone"):
      # obj_v3
      c2345 = resnet_fpn_backbone(images, config.resnet_num_block,
                                  resolution_requirement=32,
                                  tf_pad_reverse=True,
                                  use_dilations=True)
      p23456 = fpn_model(c2345, num_channel=config.fpn_num_channel, scope="fpn")

    with tf.name_scope("person_box_features"):
      # NxCx7x7 # (?, 256, 9, 5)
      person_features = self.multilevel_roi_align(p23456[:4], self.boxes)

      # [K, 9, 5, 2048]
      self.person_features = tf.transpose(person_features, perm=[0, 2, 3, 1])

  def multilevel_roi_align(self, features, rcnn_boxes):
    """ROI align pooling feature from the right level of feature."""
    config = self.config
    assert len(features) == 4
    # Reassign rcnn_boxes to levels # based on box area size
    level_ids, level_boxes = self.fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i_, boxes, featuremap in zip(itertools.count(), level_boxes, features):
      with tf.name_scope("roi_level%s" % (i_ + 2)):
        boxes_on_featuremap = boxes * (1.0 / config.anchor_strides[i_])
        all_rois.append(roi_align(featuremap, boxes_on_featuremap,
                                  config.person_h, config.person_w))

    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois

  def fpn_map_rois_to_levels(self, boxes):
    """Map rois to feature level based on box size."""
    def tf_area(boxes):
      x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
      return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.to_int32(tf.floor(4 + tf.log(
        sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))

    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]

    level_ids = [tf.reshape(x, [-1], name="roi_level%s_id" % (i_ + 2))
                 for i_, x in enumerate(level_ids)]

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes

  def get_feed_dict(self, imgfile, box):
    """Get feed dict to feed tf."""
    H = self.imgh
    W = self.imgw
    feed_dict = {}

    img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
    assert img is not None, imgfile
    img = img.astype("float32")

    # assuming box is already under H,W size image

    feed_dict[self.imgs] = img.reshape(1, H, W, 3)

    feed_dict[self.boxes] = np.array(box).reshape(-1, 4)

    return feed_dict

# --------------------------------- Future prediction


def get_activation_function(activation_function_str):
  if activation_function_str == "relu":
    return tf.nn.relu
  elif activation_function_str == "tanh":
    return tf.nn.tanh
  elif activation_function_str == "lrelu":
    return tf.nn.leaky_relu
  else:
    print("unrecognied activation function, using relu...")
    return tf.nn.relu


def get_prediction_model(model_config, sess, gpuid=0):
  with tf.device("/gpu:%s" % gpuid):
    model = PredictionModelInference(model_config, model_config.modelname)
  # this is a hack. we load multiple model into a graph, and luckily we set
  # a top_scope for prediction model so it won't be confused with person
  # appearance model
  load_model_weights(model_config.model_path, sess, top_scope="person_pred")
  return model


class Dataset(object):
  """Class for batching during training and testing."""

  def __init__(self, data, shared_data=None, config=None):
    self.data = data
    self.valid_idxs = range(self.get_data_size())
    self.num_examples = len(self.valid_idxs)
    self.config = config
    self.shared_data = shared_data

  def get_data_size(self):
    return len(self.data["obs_traj"])

  def get_by_idxs(self, idxs):
    out = collections.defaultdict(list)
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs)
    return out

  def get_batches(self, batch_size):
    """Iterator to get batches.

    Args:
      batch_size: batch size.

    Yields:
      Dataset object.
    """

    num_batches_per_epoch = int(
        math.ceil(self.num_examples / float(batch_size)))

    def grouped():
      return list(grouper(self.valid_idxs, batch_size))

    # all batches idxs from multiple epochs
    batch_idxs_iter = itertools.chain.from_iterable(
        grouped() for _ in range(1))
    for _ in range(num_batches_per_epoch):
      # so in the end batch, the None will not included
      batch_idxs = tuple(i for i in next(batch_idxs_iter)
                         if i is not None)  # each batch idxs

      # so batch_idxs might not be size batch_size
      # pad with the last item
      original_batch_size = len(batch_idxs)
      if len(batch_idxs) < batch_size:
        pad = batch_idxs[-1]
        batch_idxs = tuple(
            list(batch_idxs) + [pad for i in
                                range(batch_size - len(batch_idxs))])

      # get the actual data based on idx
      batch_data = self.get_by_idxs(batch_idxs)

      config = self.config

      # assemble a scene feat from the full scene feat matrix for this batch
      oldid2newid = {}
      new_obs_scene = np.zeros((batch_size, config.traj_obs_length, 1),
                               dtype="int32")

      for i in range(len(batch_data["obs_scene"])):
        for j in range(len(batch_data["obs_scene"][i])):
          oldid = batch_data["obs_scene"][i][j][0]
          if oldid not in oldid2newid:
            oldid2newid[oldid] = len(oldid2newid)
          newid = oldid2newid[oldid]
          new_obs_scene[i, j, 0] = newid
      # get all the feature used by this mini-batch
      scene_feat = np.zeros((len(oldid2newid), config.scene_h,
                             config.scene_w, config.scene_class),
                            dtype="float32")
      for oldid in oldid2newid:
        newid = oldid2newid[oldid]
        scene_feat[newid, :, :, :] = \
            self.shared_data["scene_feat_all"][oldid, :, :, :]

      batch_data.update({
          "batch_obs_scene": new_obs_scene,
          "batch_scene_feat": scene_feat,
          "original_batch_size": original_batch_size,
      })

      yield Dataset(batch_data)


class PredictionModelInference(PredictionModel):
  """Rewrite the future prediction model for inferencing."""

  def get_feed_dict(self, batch, person_appearance_feat_path):
    """Givng a batch of data, construct the feed dict."""

    is_train = False
    # get the cap for each kind of step first
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    P = self.P

    T_in = config.obs_len
    T_pred = config.pred_len

    feed_dict = {}

    # initial all the placeholder

    traj_obs_gt = np.zeros([N, T_in, P], dtype="float")
    traj_obs_gt_mask = np.zeros([N, T_in], dtype="bool")

    # link the feed_dict
    feed_dict[self.traj_obs_gt] = traj_obs_gt
    feed_dict[self.traj_obs_gt_mask] = traj_obs_gt_mask

    # for getting pred length during test time
    traj_pred_gt_mask = np.zeros([N, T_pred], dtype="bool")
    feed_dict[self.traj_pred_gt_mask] = traj_pred_gt_mask

    # this is needed since it is in tf.conf?
    traj_pred_gt = np.zeros([N, T_pred, P], dtype="float")
    feed_dict[self.traj_pred_gt] = traj_pred_gt  # all zero when testing,

    feed_dict[self.is_train] = is_train

    data = batch.data
    # encoder features
    # ------------------------------------- xy input

    assert len(data["obs_traj_rel"]) == N

    for i, (obs_data) in enumerate(data["obs_traj_rel"]):
      for j, xy in enumerate(obs_data):
        traj_obs_gt[i, j, :] = xy
        traj_obs_gt_mask[i, j] = True
      for j in range(config.pred_len):
        # used in testing to get the prediction length
        traj_pred_gt_mask[i, j] = True
    # ---------------------------------------

    # scene input
    # the feature index
    obs_scene = np.zeros((N, T_in), dtype="int32")
    obs_scene_mask = np.zeros((N, T_in), dtype="bool")

    feed_dict[self.obs_scene] = obs_scene
    feed_dict[self.obs_scene_mask] = obs_scene_mask
    feed_dict[self.scene_feat] = data["batch_scene_feat"]

    # each bacth
    for i in range(len(data["batch_obs_scene"])):
      for j in range(len(data["batch_obs_scene"][i])):
        # it was (1) shaped
        obs_scene[i, j] = data["batch_obs_scene"][i][j][0]
        obs_scene_mask[i, j] = True

    # [N,num_scale, T] # each is int to num_grid_class
    for j, _ in enumerate(config.scene_grids):
      this_grid_label = np.zeros([N, T_in], dtype="int32")
      for i in range(len(data["obs_grid_class"])):
        this_grid_label[i, :] = data["obs_grid_class"][i][j, :]

      feed_dict[self.grid_obs_labels[j]] = this_grid_label

    # this is the h/w the bounding box is based on
    person_h = config.person_h
    person_w = config.person_w
    person_feat_dim = config.person_feat_dim

    obs_person_features = np.zeros(
        (N, T_in, person_h, person_w, person_feat_dim), dtype="float32")

    for i in range(len(data["obs_frameidx"])):
      for j in range(len(data["obs_frameidx"][i])):
        frame_idx = data["obs_frameidx"][i][j]
        person_id = data["obs_person_id"][i]
        featfile = os.path.join(
            person_appearance_feat_path, "%d_%d.npy" % (frame_idx, person_id))
        obs_person_features[i, j] = np.squeeze(
            np.load(featfile), axis=0)

    feed_dict[self.obs_person_features] = obs_person_features

    # add other boxes,
    K = self.K  # max_other boxes
    other_boxes_class = np.zeros(
        (N, T_in, K, config.num_box_class), dtype="float32")
    other_boxes = np.zeros((N, T_in, K, 4), dtype="float32")
    other_boxes_mask = np.zeros((N, T_in, K), dtype="bool")
    for i in range(len(data["obs_other_box"])):
      for j in range(len(data["obs_other_box"][i])):  # -> seq_len
        this_other_boxes = data["obs_other_box"][i][j]
        this_other_boxes_class = data["obs_other_box_class"][i][j]

        other_box_idxs = range(len(this_other_boxes))

        other_box_idxs = other_box_idxs[:K]

        # get the current person box
        this_person_x1y1x2y2 = data["obs_person_box"][i][j]  # (4)

        for k, idx in enumerate(other_box_idxs):
          other_boxes_mask[i, j, k] = True

          other_box_x1y1x2y2 = this_other_boxes[idx]

          other_boxes[i, j, k, :] = self.encode_other_boxes(
              this_person_x1y1x2y2, other_box_x1y1x2y2)
          # one-hot representation
          box_class = this_other_boxes_class[idx]
          other_boxes_class[i, j, k, box_class] = 1

    feed_dict[self.obs_other_boxes] = other_boxes
    feed_dict[self.obs_other_boxes_class] = other_boxes_class
    feed_dict[self.obs_other_boxes_mask] = other_boxes_mask

    # ----------------------------------------------------------

    # needed since it is in tf.conf, but all zero in testing
    feed_dict[self.traj_class_gt] = np.zeros((N), dtype="int32")

    return feed_dict


def relative_to_abs(rel_traj, start_pos):
  """Relative x,y to absolute x,y coordinates.

  Args:
    rel_traj: numpy array [T,2]
    start_pos: [2]
  Returns:
    abs_traj: [T,2]
  """

  # batch, seq_len, 2
  # the relative xy cumulated across time first
  displacement = np.cumsum(rel_traj, axis=0)
  abs_traj = displacement + np.array([start_pos])  # [1,2]
  return abs_traj


def grouper(lst, num):
  args = [iter(lst)]*num
  out = itertools.izip_longest(*args, fillvalue=None)
  out = list(out)
  return out


# ------------------------------------------- for visualization


def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)


def act_pred_logits_to_list(future_act_logits):
  """Given [num_act_class] logits, return the act_name->score list."""
  # remove these activity; dominated classes
  remove_acts = [
      "activity_walking",
      "activity_standing",
      "activity_carrying",
      "Transport_HeavyCarry",
      "Riding",
      "activity_running",
      "activity_crouching",
      "activity_sitting",
  ]
  actid2act_name = {activity2id[act_name]: act_name
                    for act_name in activity2id if act_name not in remove_acts}
  future_act_logits = np.array([
      future_act_logits[i]
      for i in range(len(activity2id)) if i in actid2act_name])
  future_act_probs = softmax(future_act_logits)
  future_act_names = [
      actid2act_name[i]
      for i in range(len(activity2id)) if i in actid2act_name]
  future_acts = [(future_act_names[i], float(future_act_probs[i]))
                 for i in range(len(future_act_probs))]
  future_acts.sort(key=operator.itemgetter(1), reverse=True)

  return future_acts


def get_person_box_at_frame(tracking_results, target_frame_idx):
  """Get all the box from the tracking result at frameIdx."""
  data = []
  for frame_idx, track_id, left, top, width, height in tracking_results:
    if frame_idx == target_frame_idx:
      data.append({
          "track_id": track_id,
          "bbox": [left, top, width, height]
      })
  return data


def find_pred_data(vis_data, target_frame_idx, frame_gap):
  """Find the prediction data for visualization."""

  # [frames are 0, 12, 24, 36...]
  # so we will show the same thing from frame_idx to frame_idx + margin frame
  margin = frame_gap - 1
  vis_frame_idxs = vis_data.keys()
  vis_frame_idxs.sort()

  found_frame_idx = None
  for frame_idx in vis_frame_idxs:
    if (frame_idx <= target_frame_idx) and (
        target_frame_idx <= frame_idx + margin):
      found_frame_idx = frame_idx
      break
  return found_frame_idx


def plot_traj(img, traj, color):
  """Plot a trajectory on image."""
  traj = np.array(traj, dtype="float32")
  points = zip(traj[:-1], traj[1:])

  for p1, p2 in points:
    img = cv2.arrowedLine(img, tuple(p1), tuple(p2), color=color, thickness=1,
                          line_type=cv2.LINE_AA, tipLength=0.2)

  return img


def draw_boxes(im, boxes, labels=None, colors=None, font_scale=0.6,
               font_thick=1, box_thick=1, bottom_text=False):
  """Draw boxes with labels on an image."""

  # boxes need to be x1, y1, x2, y2
  if not boxes:
    return im

  boxes = np.asarray(boxes, dtype="int")

  FONT = cv2.FONT_HERSHEY_SIMPLEX
  FONT_SCALE = font_scale

  if labels is not None:
    assert len(labels) == len(boxes)
  if colors is not None:
    assert len(boxes) == len(colors)

  im = im.copy()

  for i in range(len(boxes)):
    box = boxes[i]

    color = (218, 218, 218)
    if colors is not None:
      color = colors[i]

    lineh = 2  # for box enlarging, replace with text height if there is label
    if labels is not None:
      label = labels[i]

      # find the best placement for the text
      ((linew, lineh), _) = cv2.getTextSize(label, FONT, FONT_SCALE, font_thick)
      bottom_left = [box[0] + 1, box[1] - 0.3 * lineh]
      top_left = [box[0] + 1, box[1] - 1.3 * lineh]
      if top_left[1] < 0:  # out of image
        top_left[1] = box[3] - 1.3 * lineh
        bottom_left[1] = box[3] - 0.3 * lineh

      textbox = [int(top_left[0]), int(top_left[1]),
                 int(top_left[0] + linew), int(top_left[1] + lineh)]

      if bottom_text:
        cv2.putText(im, label, (box[0] + 2, box[3] - 4),
                    FONT, FONT_SCALE, color=color)
      else:
        cv2.putText(im, label, (textbox[0], textbox[3]),
                    FONT, FONT_SCALE, color=color)  #, lineType=cv2.LINE_AA)

    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]),
                  color=color, thickness=box_thick)
  return im


PALETTE_HEX = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6",
    "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43",
    "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601",
    "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0",
    "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B",
    "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500",
    "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C",
    "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED",
    "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9",
    "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700",
    "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837",
    "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625",
    "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC",
    "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58",
    "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200",
    "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
    "#7ED379", "#012C58"
]


def _parse_hex_color(s):
  r = int(s[1:3], 16)
  g = int(s[3:5], 16)
  b = int(s[5:7], 16)
  return (r, g, b)

COLORS = list(map(_parse_hex_color, PALETTE_HEX))


coco_obj_classes = [
    "BG",
    "Person",  # upper case to be compatable with actev class
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
coco_obj_class_to_id = {
    coco_obj_classes[i]: i for i in range(len(coco_obj_classes))}
coco_obj_id_to_class = {
    coco_obj_class_to_id[o]: o for o in coco_obj_class_to_id}

# object detection model classname to class id
obj_class_to_id = targetClass2id_new_nopo
obj_id_to_class = {obj_class_to_id[o]: o for o in obj_class_to_id}

social_distancing_violation_acts = [
    #"Talking",
    "Person_Person_Interaction",
    "Object_Transfer",
    #"Interacts",
]
