import os
import sys
import json

import tensorflow as tf
from keras import backend as K

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn.config import Config
from mrcnn import model as mrcnn_model

sess = tf.Session()
K.set_session(sess)

COCO_WEIGHTS_PATH = os.path.join('{YOUR_PATH}/mask-r-cnn/custom/balloon20200312T1813',"mask_rcnn_balloon_0029.h5")

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "balloon"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

weight_model = mrcnn_model.MaskRCNN(
    mode="inference",
    config=config,
    model_dir=COCO_WEIGHTS_PATH
)

weight_model.load_weights(
    COCO_WEIGHTS_PATH,
    by_name=True,
    exclude=[
        "mrcnn_class_logits",
        "mrcnn_bbox_fc",
        "mrcnn_bbox",
        "mrcnn_mask"]
    )

print(type(weight_model))


