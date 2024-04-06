import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'videos', 'multi', 'DayLight')
OUTPUT_DIR = os.path.join('.', 'output_videos')

def process_video(video_path, model, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_path_out = os.path.join(OUTPUT_DIR, '{}_out.mp4'.format(video_name))
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
    
    while ret:
        results = model(frame)[0]
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        out.write(frame)
        ret, frame = cap.read()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


model_path = os.path.join('.', 'model', 'fifteen.pt')
model = YOLO(model_path)  


os.makedirs(OUTPUT_DIR, exist_ok=True)

for video_file in os.listdir(VIDEOS_DIR):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(VIDEOS_DIR, video_file)
        process_video(video_path, model)
