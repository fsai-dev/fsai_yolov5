import os
import sys
from pathlib import Path
import cv2
import json
import random
import string
import shutil
from tqdm import tqdm
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


class Colors:
    def __init__(self):
        hex = (
            "FF3838",
            "2C99A8",
            "FF701F",
            "6473FF",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "FF9D97",
            "00C2FF",
            "344593",
            "FFB21D",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex_to_rgb("#" + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, ind, bgr: bool = False):
        """
        Convert an index to a color code.

        Args:
            ind (int): The index to convert.
            bgr (bool, optional): Whether to return the color code in BGR format. Defaults to False.

        Returns:
            tuple: The color code in RGB or BGR format, depending on the value of `bgr`.
        """
        color_codes = self.palette[int(ind) % self.n]
        return (color_codes[2], color_codes[1], color_codes[0]) if bgr else color_codes

    @staticmethod
    def hex_to_rgb(hex_code):
        """
        Converts a hexadecimal color code to RGB format.

        Args:
            hex_code (str): The hexadecimal color code to convert.

        Returns:
            tuple: A tuple representing the RGB values in the order (R, G, B).
        """
        rgb = []
        for i in (0, 2, 4):
            rgb.append(int(hex_code[1 + i : 1 + i + 2], 16))
        return tuple(rgb)


def add_bbox(img, point1, point2, category_name, color):
    cv2.rectangle(
        img,
        point1,
        point2,
        color=color,
        thickness=2,
    )

    label = f"{category_name}"

    box_width, box_height = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[
        0
    ]  # label width, height
    outside = point1[1] - box_height - 3 >= 0  # label fits outside box
    point2 = point1[0] + box_width, (
        point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
    )
    # add bounding box text
    cv2.rectangle(img, point1, point2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        img,
        label,
        (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
        0,
        1,
        (255, 255, 255),
        thickness=2,
    )
    return img


def main():
    weights = "/home/ubuntu/data/classify/pylon_tp_fp/best.pt"
    data_yaml = "/home/ubuntu/data/classify/pylon_tp_fp/dataset.yaml"
    coco_results_json_path = (
        "/home/ubuntu/yolo-nas-output/predictions/yolov5/yolov5-predictions.json"
    )
    imgs_dir = "/home/ubuntu/test_images"
    output_dir = "/home/ubuntu/yolo-nas-output/predictions/yolov5"
    device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu

    imgsz = (224, 224)

    output_dir = os.path.join(output_dir, "classify-visuals")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    #  Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=False, data=data_yaml, fp16=False
    )
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    with open(coco_results_json_path, "r") as f:
        data = json.load(f)

    tmp_imgs_path = os.path.join(output_dir, "tmp")

    colors = Colors()

    total_detections = sum([len(img_preds) for img_preds in data])
    with tqdm(total=total_detections) as pbar:
        for img_preds in data:
            if len(img_preds) == 0:
                continue
            Path(tmp_imgs_path).mkdir(parents=True, exist_ok=True)
            img_name = img_preds[0]["image_name"]
            img_path = os.path.join(imgs_dir, img_name)
            img = cv2.imread(img_path)
            output_img = img.copy()
            for pred in img_preds:
                x1, y1, x2, y2 = (
                    pred["bbox"][0],
                    pred["bbox"][1],
                    pred["bbox"][0] + pred["bbox"][2],
                    pred["bbox"][1] + pred["bbox"][3],
                )
                crop_img = img[y1:y2, x1:x2]
                crop_img_path = os.path.join(
                    tmp_imgs_path,
                    "".join(random.choices(string.ascii_letters + string.digits, k=6))
                    + ".jpg",
                )
                cv2.imwrite(crop_img_path, crop_img)

                # Dataloader
                dataset = LoadImages(
                    path=crop_img_path,
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

                    output_img = add_bbox(
                        output_img,
                        (x1, y1),
                        (x2, y2),
                        prediction_class,
                        colors(prediction),
                    )
                pbar.update(1)

            cv2.imwrite(os.path.join(output_dir, img_name), output_img)
            shutil.rmtree(tmp_imgs_path)


if __name__ == "__main__":
    main()
