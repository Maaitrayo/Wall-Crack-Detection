import cv2
from yolo_segmentation import YOLOSegmentation

img = cv2.imread("input-Set/Cracked_01.jpg")
# img = cv2.resize(img, None, fx=0.7, fy=0.7)

# Segmentation detector
ys = YOLOSegmentation("model\crackDetect50epoch.pt")

bboxes, classes, segmentations, scores = ys.detect(img)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
    # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
    print("class id:", class_id)
    (x, y, x2, y2) = bbox
    if class_id == 0:
        cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)

        cv2.polylines(img, [seg], True, (0, 0, 255), 4)

        cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

cv2.imshow("image", img)
cv2.waitKey(0)