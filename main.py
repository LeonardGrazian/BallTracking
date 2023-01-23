
import os
import numpy as np
from skimage import io
import cv2
from datetime import datetime

import torch
from torch import nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional, ToTensor
from torch.utils.data import DataLoader

from custom_dataset import VideoImageDataset
from utils import overlaps_any, write_video

# constants
INPUT_DIR = 'data/'
OUTPUT_DIR = 'output/'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
WEIGHTS = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
BATCH_SIZE = 16
HUMAN_LABEL = 1
BALL_LABEL = 37
THRESHOLD_SCORE = 0.5


# @param: instances, list of 3-tuples containing the label, score,
#         and bounding box of detected instances
# @returns: list of bounding boxes for detected persons
def collect_person_boxes(instances):
    person_boxes = []
    for (label, score, box) in instances:
        if label == HUMAN_LABEL and score > THRESHOLD_SCORE:
            box = box.cpu().numpy()
            person_boxes.append(box)
    return person_boxes


# @param: instances, list of 3-tuples containing the label, score,
#         and bounding box of detected instances
# @param: person_boxes, list of bounding boxes for detected persons
# @returns: list of bounding boxes for detected balls that do not
#           intersect with any detected persons
def collect_ball_boxes(instances, person_boxes):
    ball_boxes = []
    for (label, score, box) in instances:
        if label == BALL_LABEL and score > THRESHOLD_SCORE:
            box = box.cpu().numpy()
            if not overlaps_any(box, person_boxes):
                ball_boxes.append(
                    (
                        int(0.5 * (box[0] + box[2])),
                        int(0.5 * (box[1] + box[3]))
                    )
                )
                # instances are ordered by descending score,
                # we want the ball box with the highest score
                break
    else:
        ball_boxes.append(None)
    return ball_boxes


def annotate_video(video_file, output_file, model, transforms):
    image_dataset = VideoImageDataset(video_file)
    image_dataloader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True
    )

    start = datetime.today()
    images = []
    boxes = []
    for batch, X in enumerate(image_dataloader):
        print('Working on batch {}...'.format(batch))
        images.extend(list(X.cpu().numpy()))

        inputs = transforms(X).to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs)
        for o in outputs:
            instances = [x for x in zip(o['labels'], o['scores'], o['boxes'])]
            person_boxes = collect_person_boxes(instances)
            boxes.extend(collect_ball_boxes(instances, person_boxes))
    elapsed = (datetime.today() - start).seconds
    print('Processed video of length {:.2}s in {}s.'.format(
        len(image_dataset) * 1.0 / image_dataset.fps,
        elapsed
    ))

    write_video(
        images,
        boxes,
        output_file,
        image_dataset.fps,
        image_dataset.width,
        image_dataset.height
    )


def annotate_videos(input_dir, output_dir):
    model = maskrcnn_resnet50_fpn(weights=WEIGHTS).to(DEVICE)
    model.eval()
    transforms = WEIGHTS.transforms()

    for file in os.listdir(input_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".mp4"):
            print('Working on video {}...'.format(filename))
            annotate_video(
                os.path.join(input_dir, filename),
                os.path.join(output_dir, filename),
                model,
                transforms
            )
            print()
    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    input_dir = INPUT_DIR
    if args.input:
        input_dir = args.input
    output_dir = OUTPUT_DIR
    if args.output:
        output_dir = args.output

    annotate_videos(input_dir, output_dir)
