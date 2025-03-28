import os

from const import OUTPUT_PATH, KITTI_PATH, RELATIVE_PATH_TO_VELODYNE
from src.plot import convert_point_cloud_to_bev


def process_directory(base_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    label_output_path = os.path.join(output_path, "labels")
    image_output_path = os.path.join(output_path, "images")
    os.makedirs(label_output_path, exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)

    for file in os.listdir(os.path.join(base_path, RELATIVE_PATH_TO_VELODYNE)):
        if file.endswith(".bin"):
            image_id = int(file.split(".")[0])
            save_path = os.path.join(image_output_path, "%06d.png" % image_id)
            label_path = os.path.join(label_output_path, "%06d.txt" % image_id)
            convert_point_cloud_to_bev(image_id, base_path, save_path, label_path, draw_boxes=False)


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    process_directory(base_path=KITTI_PATH, output_path=OUTPUT_PATH)
