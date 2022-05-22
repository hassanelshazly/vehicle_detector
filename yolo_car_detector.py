#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import os
import math
import glob
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


# ## Load yolo weights, cfg, and labels

# In[4]:


cfg_path = "./yolo/yolov3.cfg"
names_path = "./yolo/yolov3.names"
weights_path = "./yolo/yolov3.weights"

labels = open(names_path).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layers_names = net.getLayerNames()
output_layers_names = [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]


# ## Process Image

# In[21]:


def process_image(image):
  (H, W) = image.shape[:2]
  blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), crop=False, swapRB=True)
  net.setInput(blob)
  layers_output = net.forward(output_layers_names)
  boxes = []
  confidences = []
  class_ids = []

  for output in layers_output:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      if confidence > 0.85:
        box = detection[:4] * np.array([W, H, W, H])
        bx, by, bw, bh = box.astype("int")

        x = int(bx - (bw / 2))
        y = int(by - (bh / 2))

        boxes.append([x, y, bw, bh])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  idxes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold = 0.4, nms_threshold = 0.7)

  for idx in idxes:
    (x, y) = [boxes[idx][0], boxes[idx][1]]
    (w, h) = [boxes[idx][2], boxes[idx][3]]

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    text = "{}: {:.3f}".format(labels[class_ids[idx]], confidences[idx])
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

  return image


# ## Process Video

# In[28]:


input_video = "./test_videos/test_video.mp4"
output_video = './output_videos/test_video.mp4'

PYTHONFILE = False
if PYTHONFILE and len(sys.argv) > 1:
    input_video = sys.argv[1]
    output_video = "output_videos/" + input_video.split("/")[-1]

debug = True
if PYTHONFILE and len(sys.argv) > 2 and sys.argv[2] == "debug":
    debug = True

start_time = time.time()

clip = VideoFileClip(input_video)
get_ipython().run_line_magic('time', 'clip.fl_image(process_image).write_videofile(output_video, audio=False)')

print("Finished at {}".format((time.time() - start_time)/60))

