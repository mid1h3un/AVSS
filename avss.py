import tkinter as tk
import cv2
from ultralytics import YOLO
import supervision as sv
vehicle_detected=False
pothole_detected=False

def detect():
    video_path = 'test.mp4'
    cap = cv2.VideoCapture(video_path)

    model = YOLO("best.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    def run_process():
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        labels = [
            f"{'Vehicle' if class_id in range(1,6) and confidence > 0.5 else 'Pothole' if class_id == 0 and confidence > 0.6 else ''}"
            for _, confidence, class_id, _
            in detections
        ]        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("yolov8", frame)
        print(labels)
        # Update Headlight box
        if "Vehicle" in labels:
            light_canvas.config(bg="yellow")
        else:
            light_canvas.config(bg="white")
        # Update Pothole alert box
        if "Pothole" in labels:
            pothole_canvas.config(bg="red")
        else:
            pothole_canvas.config(bg="white")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
        root.after(100, run_process)

    run_process()

root = tk.Tk()
root.title("Light and Pothole Alert")

heading_label = tk.Label(root, text="AVSS", font=("Arial", 24))
heading_label.pack()

light_canvas = tk.Canvas(root, width=100, height=100, bg="white")
light_canvas.pack()

light_canvas.create_text(50, 50, text="Light")

pothole_canvas = tk.Canvas(root, width=100, height=100, bg="green")
pothole_canvas.pack()

pothole_canvas.create_text(50, 50, text="Pothole Alert")

start_button = tk.Button(root, text="Start", command=detect, font=("Arial", 14), bg="green", fg="white")
start_button.pack()

status_label = tk.Label(root, text="White : Bright light", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Yellow : Dim light", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Green : Reduce speed", font=("Arial", 10))
status_label.pack()
status_label = tk.Label(root, text="Red : Reduce speed", font=("Arial", 10))
status_label.pack()

root.geometry("400x400")
root.mainloop()
