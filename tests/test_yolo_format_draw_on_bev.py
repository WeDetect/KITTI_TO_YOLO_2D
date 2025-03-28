import os.path

import cv2

from const import OUTPUT_PATH


def draw_yolo_labels(image_path, label_path, output_path=None):
    """
    Draws rectangles on the image based on YOLO formatted labels.

    Args:
        image_path (str): Path to the input image.
        label_path (str): Path to the YOLO formatted label file.
        output_path (str, optional): If provided, the annotated image is saved to this path.

    Returns:
        image (numpy.ndarray): The image with drawn rectangles.
    """
    # Load the image using OpenCV.
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    height, width = image.shape[:2]

    # Open the label file and process each line.
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip lines that don't have exactly 5 components.
            class_id, x_center, y_center, box_width, box_height = map(float, parts)

            # Denormalize the coordinates based on image dimensions.
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            # Compute the top-left and bottom-right corners.
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw the rectangle.
            color = (0, 255, 0)  # Green color for the rectangle.
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Optionally, draw the class id near the rectangle.
            cv2.putText(image, str(int(class_id)), (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image if an output path is provided.
    if output_path:
        cv2.imwrite(output_path, image)

    return image


if __name__ == "__main__":
    IMAGE_ID = 1  # TODO: Change me

    image_path = os.path.join(OUTPUT_PATH, "kitti_bev_images_with_rect", "%06d.png" % IMAGE_ID)
    label_path = os.path.join(OUTPUT_PATH, "yolo_formatted_labels", "%06d.txt" % IMAGE_ID)

    yolo_test_images_dir = os.path.join(OUTPUT_PATH, "yolo_test_image")
    os.makedirs(yolo_test_images_dir, exist_ok=True)
    yolo_test_image_output = os.path.join(yolo_test_images_dir, "%06d.jpg" % IMAGE_ID)
    image_with_boxes = draw_yolo_labels(image_path, label_path, yolo_test_image_output)
    cv2.imshow("Annotated Image", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
