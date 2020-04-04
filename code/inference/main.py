# coding=utf-8
"""Run trajectory & activity prediction given a list of videos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import cv2
import json
import math
import copy
import operator
# this is for removing warnings of Gdk-CRITICAL **
import matplotlib
matplotlib.use('Agg')
# so here won"t have poll allocator info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
logging.getLogger("tensorflow").disabled = True

import tensorflow as tf
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Object_Detection_Tracking"))
from models import get_model as get_object_detection_model

# The next-prediction code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pred_utils import object2id
from pred_utils import activity2id

from inf_utils import mkdir
from inf_utils import get_video_meta
from inf_utils import resize_img
from inf_utils import get_scene_seg_model
from inf_utils import DeepSortTracker
from inf_utils import get_coco_list
from inf_utils import resize_seg_map
from inf_utils import get_traj_point
from inf_utils import convert_box
from inf_utils import clip_box
from inf_utils import load_obj_boxes
from inf_utils import get_nearest
from inf_utils import get_person_appearance_model
from inf_utils import get_prediction_model
from inf_utils import get_activation_function
from inf_utils import Dataset
from inf_utils import relative_to_abs
from inf_utils import find_past_trajs

from inf_utils import COLORS
from inf_utils import get_person_box_at_frame
from inf_utils import draw_boxes
from inf_utils import plot_traj
from inf_utils import find_pred_data
from inf_utils import social_distancing_violation_acts
from scipy.ndimage import gaussian_filter
from inf_utils import act_pred_logits_to_list

# absolute path to the repo, to get default model paths
top_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
parser = argparse.ArgumentParser()

# 1. input/output arguments.
parser.add_argument("video_list",
                    help="A file of a list of absolute path to the videos.")
parser.add_argument("output_path",
                    help="Path to store the all outputs, including "
                    "object detection, tracking, scene segmentation.")
parser.add_argument("--pred_vis_path", default=None,
                    help="If set, will generate visualization per video.")

# 2. Required model file paths
parser.add_argument(
    "--obj_model",
    default=os.path.join(top_dir, "models_and_data", "obj_model.pb"),
    help="Path to object detection model (.pb)")
parser.add_argument(
    "--scene_seg_model",
    default=os.path.join(top_dir, "models_and_data", "scene_seg.pb"),
    help="Path to scene segmentation model (.pb)")
parser.add_argument(
    "--person_appearance_model",
    default=os.path.join(top_dir, "models_and_data", "obj_v3_model"),
    help="Path to person appearance model")
parser.add_argument(
    "--prediction_model",
    default=os.path.join(top_dir, "models_and_data",
                         "next-models_nokp/actev_single_model/model/00/best/"),
    help="Path to future prediction model")
parser.add_argument(
    "--scene_id2name",
    default=os.path.join(top_dir, "models_and_data",
                         "scene36_64_id2name_top10.json"))

parser.add_argument("--dist_thresh", default=150., type=float,
                    help="Pixel distances to be considered within 6 feet."
                         " Ideally we would need to know camera parameters.")

# 3. intermediate output paths
parser.add_argument("--inter_frame_path", default="video_frames",
                    help="Under output_path, folder name to"
                    " extract the frames.")
parser.add_argument("--inter_obj_box_path", default="objs",
                    help="Under output_path, folder name to "
                    "save the intermediate object boxes in json.")
parser.add_argument("--inter_track_path", default="tracks",
                    help="Under output_path, folder name for "
                    "track file per video, MOT format.")
parser.add_argument("--inter_person_appearance_path",
                    default="person_appearance",
                    help="Under output_path, folder name for "
                    "person appearance feature per video per frame per person.")
parser.add_argument("--inter_scene_seg_path", default="scene_seg",
                    help="Under output_path, folder name for scen")
parser.add_argument("--inter_future_pred_path", default="future_pred",
                    help="Under output_path, folder name for "
                         "future prediction results.")

# 4. Runtime hardware parameters
parser.add_argument("--gpuid", default=0, type=int,
                    help="Currently only support running on single GPU.")

# 5. hyper-parameters

# 5.1 Resize the videos to this HxW, note all intermedia output will be in this
# scale.
parser.add_argument("--max_size", default=1920, type=int,
                    help="resize the video frames to..")
parser.add_argument("--shorter_edge_size", default=1080, type=int,
                    help="resize the video frames to..")

# 5.2 scene seg
parser.add_argument("--scene_seg_every", default=30, type=int,
                    help="Run scene segmentation every k frame to save compute")
parser.add_argument("--scene_seg_down_rate", default=8.0, type=float,
                    help="down-size how many times. We use 8 in the paper.")

# 5.3 object detection & tracking
parser.add_argument("--obj_model_isnot_coco", action="store_true",
                    help="obj model uses coco trained instead of actev.")
parser.add_argument("--obj_isnot_partial_model", action="store_true")
parser.add_argument("--obj_min_conf", default=0.6, type=float,
                    help="during tracking, object box with score lower than"
                    "this will be ignored.")
parser.add_argument("--obj_frame_gap", default=6, type=int,
                    help="run object detection on every k frames")

# 5.4 trajectory
parser.add_argument("--traj_drop_frame", default=12, type=int,
                    help="drop how many frame for trajectory, in the paper we "
                    "use 30 fps video and drop 12 for every frame so 2.5 fps.")
parser.add_argument("--traj_obs_length", default=8, type=int,
                    help="Observation timestep for trajectory prediction.")
parser.add_argument("--traj_pred_length", default=12, type=int,
                    help="Predict how many future timestep")

# 5.5 other box
parser.add_argument("--other_box_topk", default=15, type=int,
                    help="How many other boxes per frame to consider.")

# 5.6. person appearance feature
parser.add_argument("--person_h", default=9, type=int)
parser.add_argument("--person_w", default=5, type=int)

# 5.7 prediction model
parser.add_argument("--is_baseline", action="store_true",
                    help="baseline LSTM only model for comparison")

parser.add_argument("--pred_batch_size", default=64, type=int)
parser.add_argument("--pred_traj_emb_size", type=int, default=128)
parser.add_argument("--pred_enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--pred_dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--pred_activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")

parser.add_argument("--pred_scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--pred_scene_conv_dim", default=64, type=int)


parser.add_argument("--pred_box_emb_size", type=int, default=64)

parser.add_argument("--use_smoothing", action="store_true",
                    help="trajectory smoothing")
parser.add_argument("--smoothing_past_steps", default=4, type=int,
                    help="look how many past steps to smooth")

parser.add_argument("--use_homography", action="store_true",
                    help="use homography transformation for trajectory.")
parser.add_argument("--homography_list", default=None,
                    help="one txt file per video.")

if __name__ == "__main__":
  args = parser.parse_args()

  args.obj_model_iscoco = not args.obj_model_isnot_coco
  args.obj_is_partial_model = not args.obj_isnot_partial_model

  video_files = [line.strip() for line in open(args.video_list).readlines()]

  # compute the grid centers first
  args.scene_grid_strides = (2, 4)
  args.num_scene_grid = len(args.scene_grid_strides)

  args.scene_grids = []
  # the following is consistent with tensorflow conv2d when given odd input
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  # Get the center point for each scale's each grid
  args.scene_grid_centers = []
  for h, w in args.scene_grids:
    h_gap, w_gap = args.shorter_edge_size*1.0/h, args.max_size*1.0/w
    centers_x = np.cumsum([w_gap for _ in range(w)]) - w_gap/2.0
    centers_y = np.cumsum([h_gap for _ in range(h)]) - h_gap/2.0
    centers_xx = np.tile(np.expand_dims(centers_x, axis=0), [h, 1])
    centers_yy = np.tile(np.expand_dims(centers_y, axis=1), [1, w])
    centers = np.stack((centers_xx, centers_yy), axis=-1)  # [H,W,2]
    args.scene_grid_centers.append(centers)

  with open(args.scene_id2name, "r") as f:
    scene_id2name = json.load(f)  # {"oldid2new":,"id2name":}
  scene_oldid2new = scene_id2name["oldid2new"]
  scene_oldid2new = {
      int(oldi): scene_oldid2new[oldi] for oldi in scene_oldid2new}
  # for background class or other class that we ignored
  assert 0 not in scene_oldid2new
  scene_oldid2new[0] = 0
  total_scene_class = len(scene_oldid2new)
  scene_id2name = scene_id2name["id2name"]
  scene_id2name[0] = "BG"
  assert len(scene_oldid2new) == len(scene_id2name)

  mkdir(args.output_path)

  # Load all model weights.
  # Assuming they can fit into one GPU
  print("Loading obj model...")
  obj_model = get_object_detection_model(argparse.Namespace(
      load_from=args.obj_model, is_load_from_pb=True, add_mask=False),
                                         gpuid=args.gpuid)
  print("obj model loaded.")

  print("Loading scene seg model...")
  scene_seg_model = get_scene_seg_model(args.scene_seg_model, gpuid=args.gpuid)
  print("scene seg model loaded.")

  # Create some intermedia data folder
  track_path = os.path.join(args.output_path,
                            args.inter_track_path)
  mkdir(track_path)
  future_pred_path = os.path.join(args.output_path,
                                  args.inter_future_pred_path)
  mkdir(future_pred_path)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (",".join([
      "%s" % i for i in [args.gpuid]]))

  with tf.Session(config=tfconfig) as sess:
    # TODO(junweil): make obj model and person appearance model the same model
    # TODO(junweil): pack all the model into frozen graph; so we don't need a
    # top variable scope for each model
    print("Loading person appearance model...")
    person_appearance_config = argparse.Namespace(
        person_h=args.person_h,
        person_w=args.person_w,
        imgh=args.shorter_edge_size,
        imgw=args.max_size,
        resnet_num_block=[3, 4, 23, 3],
        fpn_num_channel=256,
        anchor_strides=(4, 8, 16, 32, 64),
        model_path=args.person_appearance_model)
    person_appearance_model = get_person_appearance_model(
        person_appearance_config, sess, gpuid=args.gpuid)
    print("person appearance model loaded.")

    print("Loading prediction model...")
    prediction_model_config = argparse.Namespace(
        modelname="model",
        batch_size=args.pred_batch_size,
        kp_size=17,

        scene_conv_kernel=args.pred_scene_conv_kernel,
        scene_conv_dim=args.pred_scene_conv_dim,
        pool_scale_idx=0,
        scene_h=args.scene_h,
        scene_w=args.scene_w,
        scene_class=args.scene_class,

        max_other=args.other_box_topk,
        obs_len=args.traj_obs_length,
        num_box_class=len(object2id),
        num_act=len(activity2id),
        pred_len=args.traj_pred_length,
        emb_size=args.pred_traj_emb_size,
        enc_hidden_size=args.pred_enc_hidden_size,
        dec_hidden_size=args.pred_dec_hidden_size,
        activation_func=get_activation_function(args.pred_activation_func),
        multi_decoder=True,
        person_h=args.person_h,
        person_w=args.person_w,
        person_feat_dim=256,
        keep_prob=1.0,
        add_kp=False,  # currently not using person keypoint
        add_other=True,
        add_person_scene=True,
        no_focal=False,
        add_grid=True,
        add_person_appearance=True,
        add_activity=True,
        use_abs_coor=False,
        inferencing_only=True,
        traj_cats=[
            ["static", 0],
            ["mov", 1],
        ],
        box_emb_size=args.pred_box_emb_size,
        scene_grids=args.scene_grids,
        model_path=args.prediction_model)

    if args.use_homography:
      prediction_model_config.multi_decoder = False

    if args.is_baseline:
      prediction_model_config.multi_decoder = False
      prediction_model_config.add_other = False
      prediction_model_config.no_focal = True
      prediction_model_config.add_grid = False
      prediction_model_config.add_person_appearance = False

    prediction_model = get_prediction_model(prediction_model_config, sess,
                                            gpuid=args.gpuid)
    print("prediction model loaded.")

    videoname_to_h_matrix = {}
    if args.use_homography:
      h_matrix_files = [line.strip()
                        for line in open(args.homography_list).readlines()]
      assert len(h_matrix_files) == len(video_files)
      # assume image to world matrix
      for h_matrix_file in h_matrix_files:
        h_matrix = []
        with open(h_matrix_file, "r") as f:
          for line in f:
            h_matrix.append(line.strip().split(","))
        h_matrix = np.array(h_matrix, dtype="float")

        video_name = os.path.splitext(os.path.basename(h_matrix_file))[0]
        _, h_inv = cv2.invert(h_matrix)
        videoname_to_h_matrix[video_name] = (h_matrix, h_inv)


    for video_file in tqdm(video_files):
      video_name = os.path.splitext(os.path.basename(video_file))[0]

      # ------------- 1. Get video frames
      tqdm.write("1. Getting video frames...")
      try:
        vcap = cv2.VideoCapture(video_file)
        if not vcap.isOpened():
          raise ValueError("Cannot open %s" % video_file)
      except ValueError as e:
        # skipping this video
        tqdm.write("Skipping %s due to %s" % (video_file, e))
        continue

      video_framepath = os.path.join(args.output_path,
                                     args.inter_frame_path,
                                     video_name)
      mkdir(video_framepath)

      video_meta = get_video_meta(vcap)

      cur_frame = 0
      while cur_frame < video_meta["frame_count"]:
        suc, frame = vcap.read()

        if not suc:
          break
        # if cur_frame == 300:  # for debugging
        # break

        frame = frame.astype("float32")

        # force to be (max_size, shorter_edge_size), not keeping ratio
        resized_frame = resize_img(frame, args.max_size, args.shorter_edge_size,
                                   force=True)

        cv2.imwrite(os.path.join(video_framepath, "%s_F_%08d.jpg" % (
            video_name, cur_frame)), resized_frame)

        cur_frame += 1

      if cur_frame != video_meta["frame_count"]:
        tqdm.write("Warning, video %s got %s frames instead of %s" % (
            video_file, cur_frame, video_meta["frame_count"]))

      # Use the actual extracted frame count for the rest
      video_meta["actual_frame_count"] = cur_frame
      tqdm.write("\tDone.")
      # ----------------- 2. Object detection & tracking
      tqdm.write("2. Object detection & tracking...")

      obj_box_path = os.path.join(args.output_path,
                                  args.inter_obj_box_path, video_name)
      mkdir(obj_box_path)

      tracker = DeepSortTracker(track_obj="Person", metric="cosine",
                                max_cosine_dist=0.5, min_detection_height=0,
                                min_confidence=args.obj_min_conf,
                                nms_max_overlap=0.85,
                                nn_budget=5, frame_gap=args.obj_frame_gap,
                                is_coco_class=args.obj_model_iscoco,
                                is_partial_model=args.obj_is_partial_model)
      for frame_idx in tqdm(range(
            0, video_meta["actual_frame_count"], args.obj_frame_gap)):
        frame_file = os.path.join(video_framepath, "%s_F_%08d.jpg" % (
            video_name, frame_idx))

        frame_data = cv2.imread(frame_file)
        frame_data = frame_data.astype("float32")

        feed_dict = obj_model.get_feed_dict_forward(frame_data)

        model_output = [obj_model.final_boxes, obj_model.final_labels,
                        obj_model.final_probs, obj_model.fpn_box_feat]
        boxes, labels, probs, box_feats = sess.run(model_output,
                                                   feed_dict=feed_dict)

        tracker.track(boxes, labels, probs, box_feats, frame_idx)

        coco_box_list = get_coco_list(boxes, labels, probs,
                                      is_coco_class=args.obj_model_iscoco)
        # save object boxes results
        obj_output_file = os.path.join(
            obj_box_path, "%s_F_%08d.json" % (video_name, frame_idx))
        with open(obj_output_file, "w") as f:
          json.dump(coco_box_list, f)

      # [frameIdx, track_id, left, top, width, height]
      tracking_results = tracker.finalize()  # will be used in step 4
      # save tracking results in MOT format
      track_output_file = os.path.join(track_path, "%s.txt" % video_name)
      with open(track_output_file, "w") as f:
        for tracking_result in tracking_results:
          f.writelines("%d,%d,%.4f,%.4f,%.4f,%.4f,1,-1,-1,-1\n" % (
              tracking_result[0], tracking_result[1], tracking_result[2],
              tracking_result[3], tracking_result[4], tracking_result[5]))

      tqdm.write("\tDone.")
      # ----------------- 3. scene segmentation feature
      tqdm.write("3. Scene segmentation...")

      scene_seg_path = os.path.join(args.output_path,
                                    args.inter_scene_seg_path, video_name)
      mkdir(scene_seg_path)

      # record the frame indexes that we run scene seg
      scene_seg_frame_indexes = []
      for frame_idx in tqdm(range(0, video_meta["actual_frame_count"],
                                  args.scene_seg_every)):
        frame_file = os.path.join(video_framepath, "%s_F_%08d.jpg" % (
            video_name, frame_idx))

        feed_dict = scene_seg_model.get_feed_dict_forward(frame_file)

        model_output = [scene_seg_model.output_tensor]
        seg_map, = sess.run(model_output, feed_dict=feed_dict)
        seg_map = seg_map[0]  # single image input

        seg_map = resize_seg_map(seg_map, args.scene_seg_down_rate)

        # save the seg map feature
        scene_seg_out_file = os.path.join(
            scene_seg_path, "%s_F_%08d.npy" % (video_name, frame_idx))
        np.save(scene_seg_out_file, seg_map)

        scene_seg_frame_indexes.append(frame_idx)

      tqdm.write("\tDone.")
      # ----------------- 4. Get the trajectory & other observation inputs.
      tqdm.write("4. Get observations...")

      frames_to_keep = range(0, video_meta["actual_frame_count"],
                             args.traj_drop_frame)

      # these are the observation data we will use for prediction
      # frameIdx_personId will be the key for each feature
      # [frameIdx, person_id, traj_x, traj_y, x, y, w, h]
      # need to clip box, some times tracking boxes outside image size
      traj_data = [one[0:2] + get_traj_point(
          clip_box(one[2:], args.max_size, args.shorter_edge_size)) +
                   convert_box(clip_box(one[2:], args.max_size,
                                        args.shorter_edge_size))
                   for one in tracking_results if one[0] in frames_to_keep]
      if not traj_data:
        print("%s video has no track detected." % video_name)
        continue

      if args.use_homography:
        assert video_name in videoname_to_h_matrix
        h_matrix = videoname_to_h_matrix[video_name][0]
        new_traj_data = []
        for f1, f2, img_x, img_y, d1, d2, d3, d4 in traj_data:

          world_x, world_y, world_z = np.tensordot(
              h_matrix, np.array([img_x, img_y, 1]), axes=1)
          world_x /= world_z
          world_y /= world_z

          new_traj_data.append([f1, f2, world_x, world_y, d1, d2, d3, d4])
        traj_data = new_traj_data


      traj_data = np.array(traj_data, dtype="float32")

      # trajectory input
      seq_list = []  # [N, obs_len, 2], N is frames*person_per_frame
      seq_list_rel = []  # relative coordinates
      num_person_in_start_frame = []  # how many people at each frame

      seq_frameidx_list = []  # [N, seq_len]

      # which scene seg feature to use
      seq_scene_frameidx_list = []  # [N, seq_len]

      seq_grid_class_list = []  # [N, strides, seq_len]
      seq_grid_target_list = []  # [N, strides, seq_len, 2]

      person_box_list = []  # [N, seq_len, 4]
      # the other boxes in the last observed frame
      # [N,1] a list of variable number of boxes
      other_box_seq_list = []
      # [N,1] # a list of variable number of boxes classes
      other_box_class_seq_list = []
      person_id_list = []  # [N]

      scene_list = []  # [N, seq_len, 1] # only the frame feature id

      # frame by frame get all the features
      frames = np.unique(traj_data[:, 0]).tolist()
      frame_data = []  # group each frame's person
      for frame_idx in frames:
        frame_data.append(traj_data[frame_idx == traj_data[:, 0], :])

      # frameIdx_personid => box, need this to extract appearance feature
      person_key2box = {}
      obj_box_cache = {}  # top object boxes given frameIdx
      scene_feat_frame2idx = {}  # the original frameidx to an incremental id

      for idx, frame_idx in enumerate(frames):
        # all persons from idx to idx + obs_length
        # note it is idx, the timestep index, not original actual frame_idx
        cur_seq_data = np.concatenate(
            frame_data[idx:idx + args.traj_obs_length], axis=0)
        # [K] # all person Id in this sequence frames [8 timestep]
        persons_in_cur_seq = np.unique(cur_seq_data[:, 1])
        # will filter out person with not enough time step
        num_person_in_cur_seq = len(persons_in_cur_seq)

        # fill in the inputs
        # 1. trajectory
        # [K, obs_len, 2] # x,y for all person sequence, starting at idx frame
        cur_seq = np.zeros((num_person_in_cur_seq, args.traj_obs_length, 2),
                           dtype="float32")
        # relative x,y for training
        cur_seq_rel = np.zeros((num_person_in_cur_seq, args.traj_obs_length, 2),
                               dtype="float32")

        # 2. grid class and regression during the observation
        cur_seq_grids_class = np.zeros(
            (num_person_in_cur_seq, args.num_scene_grid, args.traj_obs_length),
            dtype="int32")
        cur_seq_grids_target = np.zeros(
            (num_person_in_cur_seq, args.num_scene_grid,
             args.traj_obs_length, 2),
            dtype="float32")

        # 3. person box
        person_box = np.zeros((num_person_in_cur_seq, args.traj_obs_length, 4),
                              dtype="float32")

        # 4. other box
        other_box = []
        other_box_class = []

        # 5. original frame idx
        cur_seq_frame = np.zeros((num_person_in_cur_seq, args.traj_obs_length),
                                 dtype="int32")

        # 6. scene feature frame idx
        # since we don't extract scene feature every frame,
        # we need to map current frame to scene feature frame
        cur_scene_frame = np.zeros(
            (num_person_in_cur_seq, args.traj_obs_length, 1), dtype="int32")

        # 7. person_id; need this at runtime to get person key to get feature
        cur_person_id = np.zeros((num_person_in_cur_seq), dtype="int32")

        count_person = 0

        for person_id in persons_in_cur_seq:

          # [seq_len, 8] # [frameIdx, person_id, traj_x, traj_y, x, y, w, h]
          cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

          if len(cur_person_seq) != args.traj_obs_length:
            # skipping the sequence not fully cover in this frames
            continue

          # the original frame Idx
          frame_idxs = frames[idx:idx + args.traj_obs_length]
          # record the frame
          cur_seq_frame[count_person, :] = frame_idxs
          # the scene feature mapping
          for i, fi in enumerate(frame_idxs):
            scene_frame_idx = get_nearest(scene_seg_frame_indexes, fi)
            if scene_frame_idx not in scene_feat_frame2idx:
              scene_feat_frame2idx[scene_frame_idx] = len(scene_feat_frame2idx)
            feati = scene_feat_frame2idx[scene_frame_idx]
            cur_scene_frame[count_person, i, 0] = feati
            # so we could put scene feature into a matrix and get it by [feati]

          # person box & other boxes
          person_box[count_person, :, :] = cur_person_seq[:, 4:]
          cur_person_id[count_person] = person_id
          # remember the person boxes & get other boxes
          this_other_box = []
          this_other_box_class = []
          for i, this_frame_idx in enumerate(frame_idxs):
            pbox = person_box[count_person, i, :]
            key = (this_frame_idx, person_id)
            person_key2box[key] = pbox

            # other box need to load from previous output if not cached
            if key not in obj_box_cache:
              obj_boxes = load_obj_boxes(os.path.join(
                  obj_box_path, "%s_F_%08d.json" % (
                      video_name, this_frame_idx)),
                                         args.other_box_topk,
                                         object2id)
              obj_box_cache[key] = (
                  [one["bbox"] for one in obj_boxes],
                  [object2id[one["cat_name"]] for one in obj_boxes])
            # a list of [4]
            this_other_box.append(obj_box_cache[key][0])
            # a list of [1]
            this_other_box_class.append(obj_box_cache[key][1])
          other_box.append(this_other_box)
          other_box_class.append(this_other_box_class)

          # [seq_len, 2]
          cur_person_seq = cur_person_seq[:, 2:4]
          cur_person_seq_rel = np.zeros_like(cur_person_seq)

          # first frame is zeros x,y
          cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - \
              cur_person_seq[:-1, :]

          cur_seq[count_person, :, :] = cur_person_seq
          cur_seq_rel[count_person, :, :] = cur_person_seq_rel

          # grid classification observation

          # get the grid classification label based on (x,y)
          # grid centers: [H,W,2]
          for i, (center, (h, w)) in enumerate(zip(
              args.scene_grid_centers, args.scene_grids)):

            # grid classification
            h_gap, w_gap = args.shorter_edge_size * 1.0 / h, \
                args.max_size * 1.0 / w
            x_indexes = np.ceil(cur_person_seq[:, 0] / w_gap)  # [seq_len]
            y_indexes = np.ceil(cur_person_seq[:, 1] / h_gap)  # [seq_len]
            x_indexes = np.asarray(x_indexes, dtype="int")
            y_indexes = np.asarray(y_indexes, dtype="int")

            # ceil(0.0) = 0.0, we need
            x_indexes[x_indexes == 0] = 1
            y_indexes[y_indexes == 0] = 1
            x_indexes = x_indexes - 1
            y_indexes = y_indexes - 1

            one_hot = np.zeros((args.traj_obs_length, h, w), dtype="uint8")

            one_hot[range(args.traj_obs_length), y_indexes, x_indexes] = 1
            one_hot_flat = one_hot.reshape((args.traj_obs_length, -1))
            classes = np.argmax(one_hot_flat, axis=1)  # [seq_len]
            cur_seq_grids_class[count_person, i, :] = classes

            # grid regression
            # tile current person seq xy
            cur_person_seq_tile = np.tile(np.expand_dims(np.expand_dims(
                cur_person_seq, axis=1), axis=1), [1, h, w, 1])
            # tile center [seq_len,h,w,2]
            center_tile = np.tile(np.expand_dims(
                center, axis=0), [args.traj_obs_length, 1, 1, 1])
            # grid_center + target -> actual xy
            all_target = cur_person_seq_tile - center_tile  # [seq_len,h,w,2]
            # only save the one grid
            cur_seq_grids_target[count_person, i, :, :] = \
                all_target[one_hot.astype("bool"), :]

          count_person += 1

        # save per frame data into the a list
        num_person_in_start_frame.append(count_person)
        # only count_person data is preserved
        seq_list.append(cur_seq[:count_person])
        seq_list_rel.append(cur_seq_rel[:count_person])

        seq_frameidx_list.append(cur_seq_frame[:count_person])

        # scene feature mapping
        seq_scene_frameidx_list.append(cur_scene_frame[:count_person])

        person_box_list.append(person_box[:count_person])
        person_id_list.append(cur_person_id[:count_person])

        # other_box: [count_person, seqlen, K, 4] but python list,
        # K is variable length
        other_box_seq_list.extend(other_box)
        other_box_class_seq_list.extend(other_box_class)

        # scene feature
        scene_list.append(cur_scene_frame[:count_person])

        # grid classification and regression
        seq_grid_class_list.append(cur_seq_grids_class[:count_person])
        seq_grid_target_list.append(cur_seq_grids_target[:count_person])

      # put the per frame data list into a trajectory list
      num_seq = len(seq_list)  # total number of frames across all videos

      # 1. trajectory
      # [N*K, seq_len, 2]
      # N is num_frame for each video, K is num_person in each frame
      obs_traj = np.concatenate(seq_list, axis=0)
      obs_traj_rel = np.concatenate(seq_list_rel, axis=0)
      obs_frameidx = np.concatenate(seq_frameidx_list, axis=0)

      # the starting idx for each frame in the N*K list,
      # useful to get all person in one frame
      # [num_frame, 2]
      # currently not used. Used in Social GAN so they could pool all traj
      # at the same frame
      cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
      obs_start_end = np.array([
          (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
      ], dtype="int")

      # 2. person box
      obs_person_box = np.concatenate(person_box_list, axis=0)
      obs_person_id = np.concatenate(person_id_list, axis=0)

      # 3. other object box
      # other_box_seq_list a list  [N, count_person, seqlen, K, 4],
      # K is variable length
      other_box_seq_list = np.asarray(
          other_box_seq_list)  # [N*K,seqlen] list type?
      other_box_class_seq_list = np.asarray(
          other_box_class_seq_list)  # [N*K,seqlen] list type?

      # 4. grid
      obs_grid_class = np.concatenate(seq_grid_class_list, axis=0)
      obs_grid_target = np.concatenate(seq_grid_target_list, axis=0)

      # 5. scene feature load into one big matrix
      # [N*K, seq_len]
      obs_scene = np.concatenate(scene_list, axis=0)
      # stack all the feature into one big matrix
      # all frames in all videos # now it is jus the unique feature frame
      total_frames = len(scene_feat_frame2idx)
      scene_feat_final_shape = (total_frames, args.scene_h,
                                args.scene_w, total_scene_class)
      # [K, 36, 64, 11]
      tqdm.write("\tinitializing big scene feature matrix : %s.." % list(
          scene_feat_final_shape))
      # each class will be a mask
      scene_feat_all = np.zeros(scene_feat_final_shape, dtype="uint8")
      tqdm.write("\tDone.")
      tqdm.write("\tmaking mask scene feature...")
      for frame_idx in tqdm(scene_feat_frame2idx, ascii=True):

        scene_feat_file = os.path.join(
            scene_seg_path, "%s_F_%08d.npy" % (video_name, frame_idx))
        scene_feat = np.load(scene_feat_file)  # [H, W]

        feati = scene_feat_frame2idx[frame_idx]

        # transform classid first
        new_scene_feat = np.zeros_like(scene_feat)  # zero for background class
        for i in range(args.scene_h):
          for j in range(args.scene_w):
            # rest is ignored and all put into background
            if scene_feat[i, j] in scene_oldid2new:
              new_scene_feat[i, j] = scene_oldid2new[scene_feat[i, j]]
        # transform to masks
        this_scene_feat = np.zeros(
            (args.scene_h, args.scene_w, total_scene_class), dtype="uint8")
        # so we use the H,W to index the mask feat
        # generate the index first
        h_indexes = np.repeat(np.arange(args.scene_h), args.scene_w).reshape(
            (args.scene_h, args.scene_w))
        w_indexes = np.tile(np.arange(args.scene_w), args.scene_h).reshape(
            (args.scene_h, args.scene_w))
        this_scene_feat[h_indexes, w_indexes, new_scene_feat] = 1

        scene_feat_all[feati, :, :, :] = this_scene_feat
        del this_scene_feat
        del new_scene_feat
      tqdm.write("\t\tDone.")

      # save the data
      # all entry first dimension should be the same [N]
      obs_data = {
          "obs_traj": obs_traj,
          "obs_traj_rel": obs_traj_rel,

          "obs_person_box": obs_person_box,
          "obs_other_box": other_box_seq_list,
          "obs_other_box_class": other_box_class_seq_list,

          "obs_grid_class": obs_grid_class,
          "obs_grid_target": obs_grid_target,

          "obs_scene": obs_scene,

          # for getting person appearance feature
          "obs_frameidx": obs_frameidx,
          "obs_person_id": obs_person_id,
      }
      shared_data = {
          "scene_feat_all": scene_feat_all,
          "obs_start_end": obs_start_end,
      }

      tqdm.write("\ttotal frames with person %s, obs_traj shape:%s" % (
          num_seq, obs_traj.shape))

      tqdm.write("\tDone.")
      # ------------------- 5. get person appearance feature
      tqdm.write("5. Get person appearance feature...")

      person_appearance_path = os.path.join(
          args.output_path, args.inter_person_appearance_path,
          video_name)
      mkdir(person_appearance_path)
      person_key2box_frames = {}
      for person_key in person_key2box:
        frame_idx, person_id = person_key
        if frame_idx not in person_key2box_frames:
          person_key2box_frames[frame_idx] = []
        person_key2box_frames[frame_idx].append(person_key)
      #for person_key in tqdm(person_key2box):
      # frame_idx, person_id = person_key
      for frame_idx in tqdm(person_key2box_frames):
        frame_file = os.path.join(
            video_framepath, "%s_F_%08d.jpg" % (video_name, frame_idx))
        boxes = []
        person_ids = []
        for person_key in person_key2box_frames[frame_idx]:
          person_box = person_key2box[person_key]
          _, person_id = person_key
          boxes.append(person_box)
          person_ids.append(person_id)

        # [K, 9, 5, 256]
        person_features, = sess.run(
            [person_appearance_model.person_features],
            feed_dict=person_appearance_model.get_feed_dict(frame_file,
                                                            boxes))
        for i, person_id in enumerate(person_ids):
          person_appearance_file = os.path.join(
              person_appearance_path, "%d_%d.npy" % (frame_idx, person_id))
          person_feature = np.expand_dims(person_features[i], axis=0)
          np.save(person_appearance_file, person_feature)
      tqdm.write("\tDone.")
      # ------------------- 6. run future prediction for this video
      tqdm.write("6. Future prediction...")

      # save the prediction output
      out_data = {
          "seq_ids": [],  # start_frame_idx, end_obs_frame_idx, person_id
          "obs_traj_list": [],
          "pred_traj_list": [],
          "act_pred_list": [],
      }
      obs_dataset = Dataset(data=obs_data, shared_data=shared_data, config=args)

      num_batches = int(math.ceil(
          obs_dataset.num_examples / float(args.pred_batch_size)))
      for obs_batch in tqdm(obs_dataset.get_batches(args.pred_batch_size),
                            total=num_batches):
        feed_dict = prediction_model.get_feed_dict(
            obs_batch, person_appearance_path)

        # prediction output
        output_tensor = [
            prediction_model.traj_pred_out,
            #prediction_model.traj_class_logits,
            prediction_model.future_act_logits]

        traj_pred_out, future_act_logits = sess.run(
            output_tensor, feed_dict=feed_dict)

        N = obs_batch.data["original_batch_size"]
        traj_pred_out = traj_pred_out[:N]
        future_act_logits = future_act_logits[:N]

        for i in range(N):
          obs_traj = obs_batch.data["obs_traj"][i]  # [obs_len, 2]
          start_frame_idx = obs_batch.data["obs_frameidx"][i][0]
          end_obs_frame_idx = obs_batch.data["obs_frameidx"][i][-1]
          person_id = obs_batch.data["obs_person_id"][i]
          out_data["obs_traj_list"].append(obs_traj.tolist())
          out_data["seq_ids"].append(
              "%d_%d_%d" % (start_frame_idx, end_obs_frame_idx, person_id))

          # the output is relative coordinates
          this_pred_out = traj_pred_out[i]  # [pred_len, 2]
          # [T2,2]
          this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj[-1])
          out_data["pred_traj_list"].append(this_pred_out_abs.tolist())
          # [(act_name, logits)]
          out_data["act_pred_list"].append(
              act_pred_logits_to_list(future_act_logits[i]))

      if args.use_smoothing:
        new_out_data = {
            "seq_ids": [],  # start_frame_idx, end_obs_frame_idx, person_id
            "obs_traj_list": [],
            "pred_traj_list": [],
            "act_pred_list": [],
        }

        # get all seq_ids for each person
        personid_to_seqid = {}
        for i, seq_id in enumerate(out_data["seq_ids"]):
          start_frame_idx, end_obs_frame_idx, person_id = seq_id.split("_")
          start_frame_idx = int(start_frame_idx)
          end_obs_frame_idx = int(end_obs_frame_idx)
          if person_id not in personid_to_seqid:
            personid_to_seqid[person_id] = []
          personid_to_seqid[person_id].append((start_frame_idx, seq_id, i))

        for person_id in personid_to_seqid:
          personid_to_seqid[person_id].sort(key=operator.itemgetter(0))

        for i, seq_id in enumerate(out_data["seq_ids"]):
          start_frame_idx, end_obs_frame_idx, person_id = seq_id.split("_")
          start_frame_idx = int(start_frame_idx)
          end_obs_frame_idx = int(end_obs_frame_idx)
          past_traj_ids_steps = find_past_trajs(
              personid_to_seqid[person_id], start_frame_idx,
              args.smoothing_past_steps)
          # [(traj_idx, relative_past_step)]
          all_trajs_for_smoothing = [(out_data["pred_traj_list"][j], rel)
                                     for j, rel in past_traj_ids_steps]
          # [T, 2]
          this_pred = np.array(out_data["pred_traj_list"][i])
          if all_trajs_for_smoothing:
            for t in range(len(this_pred)):
              this_timestep_points = [this_pred[t]]  # [2]

              # find all past predicted point at this timestep
              for past_traj, rel_step in all_trajs_for_smoothing:
                this_past_traj_t = t - rel_step
                if (this_past_traj_t >= 0) and \
                   (this_past_traj_t < len(past_traj)):
                  this_timestep_points.append(past_traj[this_past_traj_t])

              this_timestep_points = np.array(this_timestep_points)  # [K, 2]
              assert len(this_timestep_points.shape) == 2, \
                     this_timestep_points.shape

              # [2]
              this_timestep_points = np.mean(this_timestep_points, axis=0)
              this_pred[t, :] = this_timestep_points

          new_out_data["seq_ids"].append(seq_id)
          new_out_data["obs_traj_list"].append(out_data["obs_traj_list"][i])
          new_out_data["act_pred_list"].append(out_data["act_pred_list"][i])
          new_out_data["pred_traj_list"].append(this_pred.tolist())

        out_data = new_out_data

      if args.use_homography:
        def world_traj_to_img_traj(traj, h_):
          new_traj = []
          for t in range(len(traj)):
            wx, wy = traj[t]
            ix, iy, z = np.tensordot(
                h_, np.array([wx, wy, 1]), axes=1)
            ix /= z
            iy /= z
            new_traj.append([ix, iy])

          return new_traj
        # transform all world traj to img traj for visualization
        h_matrix = videoname_to_h_matrix[video_name][1]
        new_out_data = {
            "seq_ids": [],  # start_frame_idx, end_obs_frame_idx, person_id
            "obs_traj_list": [],
            "pred_traj_list": [],
            "act_pred_list": [],
        }
        for i, seq_id in enumerate(out_data["seq_ids"]):

          new_out_data["seq_ids"].append(seq_id)
          new_out_data["obs_traj_list"].append(
              world_traj_to_img_traj(out_data["obs_traj_list"][i], h_matrix))
          new_out_data["act_pred_list"].append(out_data["act_pred_list"][i])
          new_out_data["pred_traj_list"].append(
              world_traj_to_img_traj(out_data["pred_traj_list"][i], h_matrix))
        out_data = new_out_data


      future_prediction_file = os.path.join(future_pred_path,
                                            "%s.json" % video_name)
      with open(future_prediction_file, "w") as f:
        json.dump(out_data, f)
      tqdm.write("\tDone.")

      # -----------------------------7. optional visualization
      if args.pred_vis_path is not None:
        tqdm.write("7. Making visualization...")

        vis_path = os.path.join(args.pred_vis_path, video_name)
        mkdir(vis_path)

        vis_data = {}  # end_obs_frame_idx ->
        # we will draw the same future prediction in between frames

        for seq_id, obs_traj, pred_traj, act_pred in zip(
            out_data["seq_ids"],
            out_data["obs_traj_list"],
            out_data["pred_traj_list"],
            out_data["act_pred_list"]):
          start_frame_idx, end_obs_frame_idx, person_id = seq_id.split("_")
          start_frame_idx = int(start_frame_idx)
          end_obs_frame_idx = int(end_obs_frame_idx)
          person_id = int(person_id)  # use this to retrieve person boxes
          if end_obs_frame_idx not in vis_data:
            vis_data[end_obs_frame_idx] = {}
          vis_data[end_obs_frame_idx][person_id] = {
              "obs_traj": obs_traj,  # [obs_len, 2]
              "pred_traj": pred_traj,  # [pred_len, 2]
              "act_pred": act_pred,  # [num_act_class]
          }

        # plot prediction and person bounding box at every frame

        # keep the same color for the same track
        color_assign = {}
        color_queue = copy.deepcopy(COLORS)
        for frame_idx in tqdm(range(video_meta["actual_frame_count"])):
          # 1. plot the tracking results
          # [frameIdx, track_id, left, top, width, height]
          person_boxes = get_person_box_at_frame(tracking_results, frame_idx)
          boxes, colors, labels = [], [], []
          track_id_to_boxes = {}
          for person_box in person_boxes:
            x, y, w, h = person_box["bbox"]
            boxes.append([x, y, x + w, y + h])
            track_id_to_boxes[int(person_box["track_id"])] = \
                [x, y, x + w, y + h]
            labels.append("# %d" % int(person_box["track_id"]))
            if person_box["track_id"] not in color_assign:
              this_color = color_queue.pop()
              color_assign[person_box["track_id"]] = this_color
              # recycle it
              color_queue.insert(0, this_color)
            #color = color_assign[person_box["track_id"]]
            color = (0, 255, 0)
            colors.append(color)

          frame_file = os.path.join(
              video_framepath, "%s_F_%08d.jpg" % (video_name, frame_idx))

          frame_data = cv2.imread(frame_file, cv2.IMREAD_COLOR)

          """ # not drawing tracking results
          frame_data = draw_boxes(frame_data, boxes, labels=labels,
                                  colors=colors, font_scale=0.6, font_thick=2,
                                  box_thick=2)
          """

          # 2. plot future prediction if there is data
          pred_frame_idx = find_pred_data(vis_data, frame_idx,
                                          args.traj_drop_frame)

          if pred_frame_idx is not None:

            # add a new layer for heatmap overlay
            new_layer = np.zeros((args.shorter_edge_size, args.max_size),
                                 dtype="float")

            pred_data = vis_data[pred_frame_idx]

            # check each person for social distancing violation
            # TODO(junweil) change to some scale based on person box height?
            dist_thresh = args.dist_thresh  # in pixels, in 1920x1080 resolution
            violated_persons = {}
            # TODO(junweil): change these with matrix mul to be more efficient
            for person_id in pred_data:
              pred_traj = np.array(pred_data[person_id]["pred_traj"])  # [T, 2]
              # check prediction minimal distance to all other persons
              for person_id2 in pred_data:
                if person_id2 != person_id:
                  # [T, 2]
                  pred_traj2 = np.array(pred_data[person_id2]["pred_traj"])
                  # [T]
                  dist = np.sum((pred_traj2 - pred_traj)**2, axis=-1)**(1./2)
                  dist = np.min(dist)
                  if dist <= dist_thresh:
                    # center point between this guy and the other guy's future
                    # location
                    center = (pred_traj[-1] + pred_traj2[-1])/2.0
                    # center point between the current location
                    center = (np.array(pred_data[person_id]["obs_traj"][-1]) + \
                        np.array(pred_data[person_id2]["obs_traj"][-1]))/2.0
                    if (person_id in track_id_to_boxes) and \
                        (person_id2 in track_id_to_boxes):
                      x1, y1, x2, y2 = track_id_to_boxes[person_id]
                      person1_center = (int((x1+x2)/2.), int((y1+y2)/2.))
                      x1, y1, x2, y2 = track_id_to_boxes[person_id2]
                      person2_center = (int((x1+x2)/2.), int((y1+y2)/2.))
                      center = (
                          (person1_center[0] + person2_center[0])/2.,
                          (person1_center[1] + person2_center[1])/2.)
                    violated_persons[person_id] = (person_id2, center)
                    break  # only one warning for now

              """
              # ignore activity prediction due to low mAP
              # 2.2 Plot the activity prediction if there are high confidence
              # a list of [act_name, prob], sorted
              act_pred = this_pred["act_pred"]

              # only plot activities of the following if they are in the top-3
              plot_only_act = social_distancing_violation_acts
              # add all act probs
              this_plot_acts = [
                  (act_name, act_prob)
                  for act_name, act_prob in act_pred
                  if act_name in plot_only_act]
              prob_sum = sum([prob for _, prob in this_plot_acts])
              """
            shown_act_label = {}  # person_id who have already warned
            for person_id in violated_persons:

              # plot person box
              """
              if person_id in track_id_to_boxes:
                boxes = [track_id_to_boxes[person_id]]
                colors = [(0, 0, 255)]
                frame_data = draw_boxes(frame_data, boxes, labels=None,
                                        colors=colors, font_scale=0.6,
                                        font_thick=2, box_thick=4)
              """
              # plot a circle
              if person_id in track_id_to_boxes:
                x1, y1, x2, y2 = track_id_to_boxes[person_id]
                person_center = (int((x1+x2)/2.), int((y1+y2)/2.))
                cv2.circle(frame_data, person_center,
                           radius=int(args.dist_thresh/2.0),
                           color=(0, 0, 255), thickness=2)

              other_person_id, center = violated_persons[person_id]
              this_pred = pred_data[person_id]
              pred_traj = this_pred["pred_traj"]

              act_labels = "Warning: Keep 6 feet apart!"
              act_label_color = (0, 0, 255)  # blue, BGR
              # add background for the text
              font_scale = 1.0
              font = cv2.FONT_HERSHEY_SIMPLEX
              font_thickness = 2
              ((linew, lineh), _) = cv2.getTextSize(
                  act_labels, font, font_scale, font_thickness)

              if (person_id not in shown_act_label) and \
                  (other_person_id not in shown_act_label):
                # show the text under the last pred location
                x, y = center
                # more to below the point so we dont obstruct
                more_h = - args.dist_thresh
                # COCO style text box
                padding = 10.  # pad the text box
                x1, y1, w, h = [
                    (x - linew/2.),
                    y + lineh/2.0 + more_h,
                    linew, lineh]
                # top left + bottom right
                textbox = np.array([x1, y1 - h, x1 + w, y1], dtype="float32")
                textbox_pad = np.array([
                    textbox[0] - padding,
                    textbox[1] - padding,
                    textbox[2] + padding,
                    textbox[3] + padding], dtype="float32")
                cv2.rectangle(frame_data,
                              (textbox_pad[0], textbox_pad[1]),
                              (textbox_pad[2], textbox_pad[3]),
                              color=(255, 255, 255),
                              thickness=-1)
                cv2.putText(frame_data, act_labels, (textbox[0], textbox[3]),
                            font, font_scale, color=(0, 0, 255),
                            thickness=font_thickness)

                # draw a circle around the possible meeting locations?
                """
                circle_center = (int(x), int(y))
                cv2.circle(frame_data, circle_center,
                           radius=int(args.dist_thresh/2.0),
                           color=(0, 0, 255), thickness=2)
                """
                shown_act_label[person_id] = 1
                shown_act_label[other_person_id] = 1

              # 2.3 plot future trajectory as heatmap
              last_obs_xy = np.array(this_pred["obs_traj"][-1])  # [2]
              last_obs_xy = last_obs_xy.reshape((1, 2))
              pred_traj = np.concatenate(
                  [last_obs_xy, this_pred["pred_traj"]], axis=0)

              num_between_line = 40
              # convert all the point into valid index
              traj_indexed = np.zeros_like(pred_traj)
              # [13, 2], make all xy along this line be 1.0
              # prevent index error
              for i, (x, y) in enumerate(pred_traj):
                x = round(x) - 1
                y = round(y) - 1
                if x < 0:
                  x = 0
                if y < 0:
                  y = 0
                if x >= args.max_size:
                  x = args.max_size -1
                if y >= args.shorter_edge_size:
                  y = args.shorter_edge_size - 1
                traj_indexed[i] = x, y

              for i, ((x1, y1), (x2, y2)) in enumerate(
                  zip(traj_indexed[:-1], traj_indexed[1:])):
                # all x,y between
                xs = np.linspace(x1, x2, num=num_between_line, endpoint=True)
                ys = np.linspace(y1, y2, num=num_between_line, endpoint=True)
                points = list(zip(xs, ys))
                for x, y in points[:-1]:
                  x = int(x)
                  y = int(y)
                  new_layer[y, x] = 1.0 + 1.6**i

              # 2.1 plot observation using yellow line
              #frame_data = plot_traj(frame_data, this_pred["obs_traj"],
              #                       (0, 255, 255))  # BGR
              # plot the prediction trajectory as a line, too
              #frame_data = plot_traj(frame_data, pred_traj,
              #                       (0, 255, 0))  # BGR

            # add gaussian filter to the trajectory prediction on new_layer
            if np.sum(new_layer) > 0:
              f_new_layer = gaussian_filter(new_layer, sigma=7)
              f_new_layer /= f_new_layer.max()
              f_new_layer = np.uint8(f_new_layer * 255)
              ret, mask = cv2.threshold(f_new_layer, 1, 255, cv2.THRESH_BINARY)
              heatmap_img = cv2.applyColorMap(f_new_layer, cv2.COLORMAP_WINTER)

              heatmap_img_masked = cv2.bitwise_and(heatmap_img, heatmap_img,
                                                   mask=mask)

              frame_data = cv2.addWeighted(frame_data, 1.0, heatmap_img_masked,
                                           0.9, 0)

          vis_file = os.path.join(
              vis_path, "%s_F_%08d.jpg" % (video_name, frame_idx))
          cv2.imwrite(vis_file, frame_data)

        tqdm.write("\tDone.")











