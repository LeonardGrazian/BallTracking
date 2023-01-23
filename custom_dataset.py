
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader


def process_image(image):
    assert len(image.shape) == 3
    # OpenCV uses BGR, PyTorch model expects RGB
    image = image[:, :, [2, 1, 0]]
    # channel should be first dimension for PyTorch
    image = np.moveaxis(image, 2, 0)
    return image


class VideoImageDataset(Dataset):
    def __init__(self, video_filename):
        self.video_filename = video_filename

        vidcap = cv2.VideoCapture(self.video_filename)
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        self.height, self.width, self.layers = image.shape

        self.images = []
        while success:
            self.images.append(process_image(image))
            success, image = vidcap.read()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
