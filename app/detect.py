from ultralytics import YOLO

# Load the model
model = YOLO('models/best.pt')

# Test on a sample image (use one from data/raw/images/ or any photo of people with/without helmets)
results = model('data/raw/images/hard_hat_workers0.png')  
# Show results (opens a window with boxes)
results[0].show()
print(results)  # Prints detections in terminal

# Basic check: Persons without helmets
persons = [b for b in results[0].boxes if int(b.cls) == 0]
helmets = [b for b in results[0].boxes if int(b.cls) == 1]
if len(persons) > len(helmets):
    print("ALERT: Person without helmet!")