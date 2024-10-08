import os
import sys
from pathlib import Path
import cv2
import shutil
import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import LoadImages
from utils.general import (
    Profile,
    check_img_size,
)
from utils.torch_utils import select_device
from utils.plots import imshow_cls


def main():
    cropped_imgs_dir = "/Users/hanna/Downloads/tmp/tp_fp_test_chips"
    output_dir = "/Users/hanna/Downloads/tmp/output"
    weights = "/Users/hanna/Downloads/tmp/pylon_tp_fp_model/epoch90.pt"
    data_yaml = "/Users/hanna/Downloads/tmp/pylon_tp_fp_model/dataset.yaml"
    device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz = (224, 224)

    #  Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=False, data=data_yaml, fp16=False
    )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Get all the images in the directory
    all_images = os.listdir(cropped_imgs_dir)

    # Get the image paths
    img_paths = [os.path.join(cropped_imgs_dir, img) for img in all_images]

    for img_path in img_paths:
        if (
            not img_path.endswith(".jpg")
            and not img_path.endswith(".png")
            and not img_path.endswith(".jpeg")
        ):
            continue
        print(f"Processing image: {img_path}")

        # Dataloader
        dataset = LoadImages(
            path=img_path,
            img_size=imgsz,
            transforms=classify_transforms(imgsz[0]),
            vid_stride=1,
        )

        dt = (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.Tensor(im).to(model.device)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                results = model(im)

            # Post-process
            with dt[2]:
                pred = F.softmax(results, dim=1)  # probabilities

            enum_pred = enumerate(pred)

            _, prob = next(enum_pred)
            prediction = prob.argsort(0, descending=True).tolist()[0]
            prediction_class = names[prediction]

            print(f"Prediction: {prediction_class}")

            # If prediction_class = FALSE_POSITIVE then copy to the false_positive directory
            Path(os.path.join(output_dir, "false_positives")).mkdir(
                parents=True, exist_ok=True
            )
            Path(os.path.join(output_dir, "true_positives")).mkdir(
                parents=True, exist_ok=True
            )

            if prediction_class == "FALSE_POSITIVE":
                print("Copying to false_positive directory")
                cv2.imwrite(
                    filename=os.path.join(
                        output_dir, "false_positives", img_path.split("/")[-1]
                    ),
                    img=im0s,
                )

            # If prediction_class = TRUE_POSITIVE then copy to the true_positive directory
            if prediction_class == "TRUE_POSITIVE":
                print("Copying to true_positive directory")
                cv2.imwrite(
                    filename=os.path.join(
                        output_dir, "true_positives", img_path.split("/")[-1]
                    ),
                    img=im0s,
                )


if __name__ == "__main__":
    main()
