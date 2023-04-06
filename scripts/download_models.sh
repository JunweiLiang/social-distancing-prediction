
# please run this at the top level of this repository
# Download all the models into models_and_data
# about 468 MB

# 1. object detection model
wget https://precognition.team/shares/diva_obj_detect_models/models/obj_coco_resnet50_partial_tfv1.14_1920x1080_rpn300.pb -O models_and_data/obj_model.pb
# 2. scene segmentation
wget https://precognition.team/next/data/072019_prepare_data/deeplabv3_xception_ade20k_train.pb -O models_and_data/scene_seg.pb
# 3. person appearance CNN model
wget https://precognition.team/shares/diva_obj_detect_models/models/obj_v3_model.tgz -O models_and_data/person_cnn.tgz
# 4. future prediction model
wget https://precognition.team/next/data/072019_prepare_data/next-models_nokp_072019.tgz -O models_and_data/prediction_model.tgz

# extract and delete the tar files
cd models_and_data

tar -zxvf person_cnn.tgz
rm person_cnn.tgz
tar -zxvf prediction_model.tgz
rm prediction_model.tgz

cd ..
