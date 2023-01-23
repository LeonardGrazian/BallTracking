
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# @param: image, tensor representing image of shape (3, W, H)
# @param: boxes, list of tensors of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of each box
def plot_tensor_with_boxes(image, boxes):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(np.moveaxis(image.detach().numpy(), 0, 2))
    for box in boxes:
        box = box.detach().numpy()
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


# @param: box1, numpy array of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of the box
# @param: box2, numpy array of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of the box
# @returns: True if box1 overlaps box2, False otherwise
def overlaps(box1, box2):
    return (
        (box1[0] > box2[0] and box1[0] < box2[2])
        or (box1[2] > box2[0] and box1[2] < box2[2])
    ) and (
        (box1[1] > box2[1] and box1[1] < box2[3])
        or ((box1[3] > box2[1] and box1[3] < box2[3]))
    )


# @param: box, numpy array of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of the box
# @param: boxes, list of numpy arrays of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of each box
# @returns: True if box overlaps with any other box_ in boxes
def overlaps_any(box, boxes):
    for box_ in boxes:
        if overlaps(box, box_):
            return True
    return False


# @param: images, list of numpy arrays of shape (3, width, height)
# @param: boxes, list of numpy arrays of shape (4,)
#         corresponding to (xmin, ymin, xmax, ymax) of each box
# @param: video_file, str, filepath to output video
# @param: fps, int, frames per second of output video
# @param: width, int, pixel width of output video
# @param: height, int, pixel height of output video
# @returns: None
def write_video(images, boxes, video_file, fps, width, height):
    assert len(images) == len(boxes)
    video_output = cv2.VideoWriter(
        video_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    for image, box in zip(images, boxes):
        image = np.moveaxis(image, 0, 2)
        image = image[:, :, [2, 1, 0]]
        contig_image = image.copy()
        if box is not None:
            cv2.circle(contig_image, (box[0], box[1]), 10, (255, 0, 0), 2)
        video_output.write(contig_image)
    video_output.release()
