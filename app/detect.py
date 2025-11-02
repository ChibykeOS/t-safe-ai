from ultralytics import YOLO
import cv2  # For frame handling
import numpy as np

model = YOLO('models/best.pt')  # Path from app/

def detect_and_alert(frame):
    # Run detection
    results = model(frame)[0]
    
    # Draw boxes on frame
    annotated_frame = results.plot()
    
    # Violation check (no helmet)
    boxes = results.boxes
    persons = [b for b in boxes if int(b.cls) == 0]
    helmets = [b for b in boxes if int(b.cls) == 1]
    violations = []
    for person in persons:
        person_box = person.xyxy[0].cpu().numpy()
        has_helmet = False
        for helmet in helmets:
            helmet_box = helmet.xyxy[0].cpu().numpy()
            # Rough head area IoU
            head_ymin = person_box[1]
            head_ymax = person_box[1] + 0.2 * (person_box[3] - person_box[1])
            head_box = [person_box[0], head_ymin, person_box[2], head_ymax]
            inter_xmin = max(head_box[0], helmet_box[0])
            inter_ymin = max(head_box[1], helmet_box[1])
            inter_xmax = min(head_box[2], helmet_box[2])
            inter_ymax = min(head_box[3], helmet_box[3])
            inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
            head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
            helmet_area = (helmet_box[2] - helmet_box[0]) * (helmet_box[3] - helmet_box[1])
            iou = inter_area / (head_area + helmet_area - inter_area + 1e-6)
            if iou > 0.3:
                has_helmet = True
                break
        if not has_helmet:
            violations.append("No helmet on person")
    
    return annotated_frame, violations  # Return frame with boxes, list of alerts
