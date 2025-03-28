import os

from const import KITTI_PATH, OUTPUT_PATH
from src.plot import convert_point_cloud_to_bev

IMAGE_ID = 1  # TODO: Change me to the relevant id


def test_draw_on_bev(base_path, output_path):
    """
    The function ment for self validation.
    The function will create a BEV image and draw rectangles based on the relevant KITTI formatted label.
    """
    os.makedirs(output_path, exist_ok=True)
    label_output_path = os.path.join(output_path, "yolo_formatted_labels")
    image_output_path = os.path.join(output_path, "kitti_bev_images_with_rect")
    os.makedirs(label_output_path, exist_ok=True)
    os.makedirs(image_output_path, exist_ok=True)

    save_path = os.path.join(image_output_path, "%06d.png" % IMAGE_ID)
    label_path = os.path.join(label_output_path, "%06d.txt" % IMAGE_ID)
    convert_point_cloud_to_bev(IMAGE_ID, base_path, save_path, label_path, draw_boxes=True)


if __name__ == "__main__":
    TEST_OUTPUT = os.path.join(OUTPUT_PATH, 'tests', 'output')
    test_draw_on_bev(base_path=KITTI_PATH, output_path=OUTPUT_PATH)
