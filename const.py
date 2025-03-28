import os

from dotenv import load_dotenv

load_dotenv()

X_RANGE = (-30, 30)
Y_RANGE = (-30, 30)
Z_RANGE = (-10, 10)

BEV_HEIGHT = 640
BEV_WIDTH = 640

CLASS_NAME_TO_ID = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Truck': 3,
    'Van': 4,
    'Tram': 5,
    'Misc': 6
}

RELATIVE_PATH_TO_VELODYNE = "data_object_velodyne/training/velodyne"
RELATIVE_PATH_TO_LABEL_2 = "data_object_label_2/training/label_2"
RELATIVE_PATH_TO_CALIB = "data_object_calib/training/calib"


KITTI_PATH = os.getenv("KITTI_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", os.path.join(os.getcwd(), "output"))
