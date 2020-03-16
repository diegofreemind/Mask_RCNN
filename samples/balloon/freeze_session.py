# thanks to https://www.gitmemory.com/issue/matterport/Mask_RCNN/218/466497448

import os
import sys

import keras.backend as K
import tensorflow as tf

# I needed to add this
sess = tf.Session()
K.set_session(sess)

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

from mrcnn import model as modellib
from mrcnn.config import Config

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or [] # for i in len(output_names)
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = '{YOUR_PATH}/mask-r-cnn/proto'
    tf.train.write_graph(frozen_graph, directory, name + '.pb', as_text=False)


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = "balloon"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()

MODEL_DIR = '{YOUR_PATH}/mask-r-cnn/custom/balloon20200312T1813'
H5_WEIGHT_PATH = '{YOUR_PATH}/mask_rcnn_balloon_0029.h5'
FROZEN_NAME = 'frozen_graph_ballons'

model = modellib.MaskRCNN(
    mode="inference",
    config=config,
    model_dir=MODEL_DIR)

model.load_weights(
    H5_WEIGHT_PATH,
    by_name=True,
        exclude=[
        "mrcnn_class_logits",
        "mrcnn_bbox_fc",
        "mrcnn_bbox",
        "mrcnn_mask"]
    )

# model = keras.models.load_model(input_model_path)
freeze_model(model.keras_model, FROZEN_NAME)