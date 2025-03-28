# KITTI TO YOLO (2D)
In this project we adjust the KITTI dataset to YOLO.
We convert the point cloud data to Bird Eye View (BEV) images and format the KITTI labels into YOLO format labels.

## Pre-run
1. Download KITTI dataset from https://github.com/zhulf0804/PointPillars
   1. point cloud(29GB) https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
   2. calibration files(16 MB) https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
   3. labels(5 MB) https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
2. Unpack the zip folders and save them under one folder. The file structure should look as the following:
    ```bash
   KITTI/
        data_object_calib/
            testing/
                calib/
                    000000.txt
                    ...
            training/
                calib/
                    000000.txt
                    ...
        data_object_label_2/
            training/
                label_2/
                    000000.txt
                    ...
        data_object_velodyne/
            testing/
                velodyne/
                    000000.bin
                    ...
            training/
                velodyne/
                    000000.bin
                    ...
   ```
3. Create a `.env` file and write the path to the dataset. (you can also specify the output folder path)
   ```txt
   KITTI_PATH=path/to/dataset/
   ```
4. Create a virtualenv (original ran on python3.12)
5. Install requirements 
   ```bash
   pip install -rm requirements.txt
   ```
## Run
Go to `main.py` and run the code.

## New Dataset sanity evaluation
We can evaluate and do some sanity test to our core functions in `test` folder.
1. Check out KITTI label projection on BEV image.
   1. Go to `test/test_kitti_format_draw_on_bev_image.py` file. 
   2. Run the test. You can select a point cloud using the IMAGE_ID const.
   3. Make sure the image contains a box that contains an object
2. Check out YOLO label projection on BEV image.
   1. Go to `test/test_yolo_format_draw_on_bev.py` file. 
   2. Make sure you already run test 1 (`test/test_kitti_format_draw_on_bev_image.py`)
      The test will use the BEV image from this test, so we will be able to see differences.
   3. Run the test. You can select an image using the IMAGE_ID const.
   4. Make sure the image contains a box that contains an object