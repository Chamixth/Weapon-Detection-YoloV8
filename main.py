import os
from ultralytics import YOLO
import cv2

# Paths
PHOTOS_DIR = os.path.join('.')

photo_path = os.path.join(PHOTOS_DIR, 'Brussels-gun-attacker-caught-on-CCTV-2-_jpg.rf.a616c58255d33043b870250e9463b6d3.jpg')
photo_path_out = '{}_out.jpg'.format(photo_path)

# Photo
img = cv2.imread(photo_path)
if img is None:
    print("Error: Could not open image.")
    exit()

# Initialize model
model_path = os.path.join('.', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # load a custom model
threshold = 0

# Function for drawing predictions on an image
def draw_predictions(image, predictions):
    for result in predictions.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, predictions.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Process photo
results_photo = model(img)[0]
draw_predictions(img, results_photo)
cv2.imwrite(photo_path_out, img)

# Release resources
cv2.destroyAllWindows()
