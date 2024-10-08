import cv2
import numpy as np
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.torch_utils import torch_distributed_zero_first, select_device
from utils.dataloaders import InfiniteDataLoader, seed_worker
from models.common import DetectMultiBackend
from utils.general import check_img_size

import torch
from torch.utils.data import Dataset, distributed
import torchvision.transforms as T


class ClassificationDataset(Dataset):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(
        self,
        img,
        imgsz,
    ):
        self.image = img
        self.torch_transforms = T.Compose(
            [
                self.Resize(imgsz),
                self.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def __getitem__(self, i):
        sample = self.torch_transforms(self.image)
        return sample

    def __len__(self):
        return 1

    class ToTensor:
        # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
        def __init__(self, half=False):
            super().__init__()
            self.half = half

        def __call__(self, im):  # im = np.array HWC in BGR order
            im = np.ascontiguousarray(
                im.transpose((2, 0, 1))[::-1]
            )  # HWC to CHW -> BGR to RGB -> contiguous
            im = torch.from_numpy(im)  # to torch
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0-255 to 0.0-1.0
            return im

    class Resize:
        # YOLOv5 Resize class for image preprocessing, i.e. T.Compose([Resize(size), ToTensor()])
        def __init__(self, size=224):
            super().__init__()
            self.h, self.w = (size, size) if isinstance(size, int) else size

        def __call__(self, im):  # im = np.array HWC
            # return cv2.resize(im, (self.h, self.w), interpolation=cv2.INTER_LINEAR)
            old_size = im.shape[:2]  # (height, width)

            # Check if the image is already the target size
            if old_size[0] == self.h and old_size[1] == self.w:
                return im

            # Determine the scaling factor to fit the image within the target size
            ratio = float(self.h) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            # Resize the image to fit within the target size
            resized_image = cv2.resize(im, (new_size[1], new_size[0]))

            # Calculate padding to center the resized image in the 224x224 frame
            delta_w = self.w - new_size[1]
            delta_h = self.h - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # Create padding color (black in this case, but can be changed)
            color = [0, 0, 0]  # Black padding

            # Apply padding to the resized image
            resized = cv2.copyMakeBorder(
                resized_image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=color,
            )
            return resized


def create_classification_dataloader(img, imgsz=224, batch_size=1, rank=-1, workers=1):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(img, imgsz)
    batch_size = min(batch_size, 1)
    nd = torch.cuda.device_count()
    nw = min(
        [os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers]
    )
    sampler = (
        None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=False)
    )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + int(os.getenv("RANK", -1)))
    return InfiniteDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        sampler=sampler,
        pin_memory=str(os.getenv("PIN_MEMORY", True)).lower() == "true",
        worker_init_fn=seed_worker,
        generator=generator,
    )


class YoloV5Classifier:
    def __init__(
        self,
        device,
        model_path,
        batch_size=1,
        workers=1,
        imgsz=224,
    ):
        self.model_path = model_path
        self.batch_size = batch_size
        self.workers = workers
        self.imgsz = imgsz
        self.device = device

        # Initialize the classification model
        device = select_device(self.device, batch_size=self.batch_size)

        # Load model
        self.classification_model = DetectMultiBackend(
            self.model_path, device=device, dnn=False, fp16=False
        )
        stride = self.classification_model.stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        self.classification_model.eval()

    def detect(self, img):
        dataloader = create_classification_dataloader(
            img=img,
            imgsz=self.imgsz,
            batch_size=self.batch_size,
            workers=self.workers,
        )

        # Get the preditions
        with torch.no_grad():
            for batch in dataloader:
                results = (
                    torch.max(self.classification_model(batch.to(self.device)), 1)[1]
                    .cpu()
                    .numpy()
                )
                predictions = [self.classification_model.names[i] for i in results]

        return predictions[0]
