import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
video_path_out = os.path.join(VIDEOS_DIR, 'output.mp4')


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = None

model_path = os.path.join('.', 'model', 'last.pt')


model = YOLO(model_path)  

threshold = 0.5


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    if out is None:
        out = cv2.VideoWriter(video_path_out, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
