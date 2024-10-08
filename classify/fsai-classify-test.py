import os
import sys
from pathlib import Path
import cv2
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from classify.Yolov5ClassifierModel import YoloV5Classifier


def main():
    cropped_imgs_dir = "/Users/hanna/Downloads/tmp/tp_fp_test_chips"
    output_dir = "/Users/hanna/Downloads/tmp/output"
    weights = "/Users/hanna/Downloads/tmp/pylon_tp_fp_model/epoch90.pt"
    device = "cpu"  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz = 224

    #  Load model
    classifier_model = YoloV5Classifier(device=device, model_path=weights, imgsz=imgsz)

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

        img = cv2.imread(img_path)
        prediction_class = classifier_model.detect(img)

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
            shutil.copy(img_path, os.path.join(output_dir, "false_positives"))

        # If prediction_class = TRUE_POSITIVE then copy to the true_positive directory
        if prediction_class == "TRUE_POSITIVE":
            print("Copying to true_positive directory")
            shutil.copy(img_path, os.path.join(output_dir, "true_positives"))


if __name__ == "__main__":
    main()
