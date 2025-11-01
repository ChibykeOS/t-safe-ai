from ultralytics import YOLO
import numpy as np  # For overlap calc

# Load model
model = YOLO('models/best.pt')

# Test image (change to yours)
results = model('data/raw/images/hard_hat_workers1.png')

# Show detections
results[0].show()

# Violation check
boxes = results[0].boxes
persons = [b for b in boxes if int(b.cls) == 0]  # Class 0 = person
helmets = [b for b in boxes if int(b.cls) == 1]  # Class 1 = helmet

for person in persons:
    person_box = person.xyxy[0].cpu().numpy()  # [xmin, ymin, xmax, ymax]
    has_helmet = False
    for helmet in helmets:
        helmet_box = helmet.xyxy[0].cpu().numpy()
        # Simple IoU overlap (head area rough: top 20% of person)
        head_ymin = person_box[1]
        head_ymax = person_box[1] + 0.2 * (person_box[3] - person_box[1])
        head_box = [person_box[0], head_ymin, person_box[2], head_ymax]
        # IoU calc
        inter_xmin = max(head_box[0], helmet_box[0])
        inter_ymin = max(head_box[1], helmet_box[1])
        inter_xmax = min(head_box[2], helmet_box[2])
        inter_ymax = min(head_box[3], helmet_box[3])
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        head_area = (head_box[2] - head_box[0]) * (head_box[3] - head_box[1])
        helmet_area = (helmet_box[2] - helmet_box[0]) * (helmet_box[3] - helmet_box[1])
        iou = inter_area / (head_area + helmet_area - inter_area + 1e-6)
        if iou > 0.3:  # Threshold for "on head"
            has_helmet = True
            break
    if not has_helmet:
        print("VIOLATION: Person without helmet!")

print("Detection complete.")

import requests

# Email alert (replace with your details; use app password for Gmail)
#if not has_helmet:
   # url = "https://api.mailgun.net/v3/your-domain.mailgun.org/messages"  # Free Mailgun signup, or use smtplib for Gmail
   # auth = ("api", "your-mailgun-key")
   # data = {
    #    "from": "alert@t-safe.ai",
     #   "to": "your@email.com",
      #  "subject": "Safety Violation",
      #  "text": "Person without helmet detected!"
  #  }
  #  response = requests.post(url, auth=auth, data=data)
 #   print("Alert sent!")