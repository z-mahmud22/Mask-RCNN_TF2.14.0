# Mask R-CNN for Object Detection and Instance Segmentation on TensorFlow==2.14.0 and Python 3.10.12
This is an implementation of the [Mask R-CNN](https://arxiv.org/abs/1703.06870) model which edits the original [Mask_RCNN](https://github.com/matterport/Mask_RCNN) repository (which only supports TensorFlow 1.x), so that it works with Python 3.10.12 and TensorFlow 2.14.0. This new reporsitory allows to train and test (i.e make predictions) the Mask R-CNN  model in TensorFlow 2.14.0. The Mask R-CNN model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

Compared to the source code of the old [Mask_RCNN](https://github.com/matterport/Mask_RCNN) repo, the  edits the following 2 modules:

1. `model.py`
2. `utils.py`

Apart from that, this repository uses the same training and testing code as in the old repo and similarly includes:

* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset

The [Mask-RCNN_TF2.14.0-keras2.14.0](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0-keras2.14.0) repo is tested with TensorFlow 2.14.0, Keras 2.14.0, and Python 3.10.12 for the following system specifications:

1. GPU - `GeForce RTX 3060 12GiB` , `Tesla T4 16GiB` (Google colab)
2. OS -  `Linux 5.15.120+, Ubuntu20.04, Windows 10 and Windows 11`
3. Cloud - `Google colab`
   
**Note: This repo does not support any of the available versions of TensorFlow 1.x.**

# Use the Repository Without Installation
It is not required to build the repo. It is enough to copy the `mrcnn` directory to where you are using it.

Please follow these steps to use the repo for making predictions:

1. Create a root directory (e.g. maskrcnn)
2. Copy the mrcnn directory inside the root directory. ## FIX THE mrcnn HERE!!!
3. Download the pre-trained MS COCO weights inside the root directory from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
4. Create a script for object detection and save it inside the root directory. This script is an example: samples/mrcnn-prediction.py. The next section will walkthrough this sample script.
5. Run the script.

The directory tree of the repo is as follows:
```
maskrcnn:
├───mrcnn:
├───mask_rcnn_coco.h5
└───mask-rcnn-prediction.py
```

# Code for Prediction/Inference
This sample code uses the pre-trained MS COCO weights of the Mask R-CNN model which can be downloaded from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5. The code is accessible through the samples/mrcnn-prediction.py script.

The [MS COCO](https://cocodataset.org/#home) dataset has 80 classes. There is an additional class for the background named **BG**. Thus, the total number of classes is 81. The classes names are listed in the `CLASS_NAMES` list. **DO NOT CHANGE THE ORDER OF THE CLASSES.**

After making prediction with this code: 
```
import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
# Download the mask_rcnn_coco.h5 file from this link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)

# load the input image, convert it from BGR to RGB channel
image = cv2.imread("test.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform a forward pass of the network to obtain the results
r = model.detect([image], verbose=0)

# Get the results for the first image.
r = r[0]

# Visualize the detected objects.
mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'])
```
it displays the input image by drawing the bounding boxes, masks, class labels, and prediction scores over all detected objects:
##input image here

# Getting Started
* mrcnn-prediction.py: A script for loading the pre-trained weights and making predictions using the Mask R-CNN model.

* coco_labels.txt: The class labels of the COCO dataset.

* demo.ipynb: is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images. It includes code to run object detection and instance segmentation on arbitrary images.

* train_shapes.ipynb: shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* (model.py, utils.py, config.py): These files contain the main Mask RCNN implementation.

* inspect_data.ipynb: This notebook visualizes the different pre-processing steps to prepare the training data.

* inspect_model.ipynb: This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* inspect_weights.ipynb: This notebook inspects the weights of a trained model and looks for anomalies and odd patterns.

# Step by Step Detection
To help with debugging and understanding the model, there are 3 notebooks (inspect_data.ipynb, inspect_model.ipynb, inspect_weights.ipynb) that provide a lot of visualizations and allow running the model step by step to inspect the output at each point. Here are a few examples:

## 1. Anchor Sorting and Filtering 
Visualizes every step of the first stage Region Proposal Network and displays positive and negative anchors along with anchor box refinement.
