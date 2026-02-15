import cv2
import numpy as np
import urllib.request
import requests
import time
from pathlib import Path
from ultralytics import YOLO

# URL of your ESP32-CAM hi-res endpoint
ESP32_CAM_URL = "http://172.20.10.2/cam-hi.jpg"
ESP32_BASE_URL = "http://172.20.10.2"

# Detection settings
CONF_THRESHOLD = 0.6   # confidence threshold (higher = more reliable)
FRAME_DELAY = 0.03     # ~30 FPS
DETECTION_BUFFER = 5   # require detection stable for N frames before toggling
CONFIDENCE_SMOOTHING = True  # average confidence over multiple frames

# Set up model file paths
script_dir = Path(__file__).parent
model_path = script_dir / "yolov8n.pt"

# Load YOLOv8 model for person detection
if not model_path.exists():
    print(f"Error: {model_path} not found. Please ensure yolov8n.pt is in the same directory as this script.")
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Loading YOLOv8 model from {model_path}...")
model = YOLO(str(model_path))
print("Model loaded successfully!")

# Keep track of last detection state
person_detected = False
detection_history = []  # track detections over time for smoothing
max_confidence = 0.0

def get_frame():
    """Capture a single frame from ESP32-CAM."""
    try:
        img_resp = urllib.request.urlopen(ESP32_CAM_URL, timeout=2)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("Failed to get frame:", e)
        return None

def detect_person(frame):
    """Return True if person is detected above confidence threshold."""
    global max_confidence
    
    # Run YOLOv8 inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    
    max_conf = 0.0
    person_found = False
    
    # Check if person (class 0) is detected
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Class 0 is 'person' in COCO dataset
                if class_id == 0:
                    max_conf = max(max_conf, confidence)
                    if confidence > CONF_THRESHOLD:
                        person_found = True
                        # Draw bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw status and confidence info
    status_color = (0, 255, 0) if person_found else (0, 0, 255)
    status_text = f"PERSON DETECTED (conf: {max_conf:.2f})" if person_found else f"No person (best: {max_conf:.2f})"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Show smoothing status
    cv2.putText(frame, f"Detection buffer: {len(detection_history)}/{DETECTION_BUFFER}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    max_confidence = max_conf
    return person_found

def trigger_esp32(person_present):
    """Call ESP32 endpoints to toggle detection pin with debouncing."""
    global person_detected, detection_history
    
    # Add to history
    detection_history.append(person_present)
    
    # Keep only recent history
    if len(detection_history) > DETECTION_BUFFER:
        detection_history.pop(0)
    
    # Only trigger if we have a stable detection
    stable_person = sum(detection_history) >= (DETECTION_BUFFER * 0.6)  # 60% consensus
    
    try:
        if stable_person and not person_detected:
            requests.get(f"{ESP32_BASE_URL}/person-detected", timeout=3.0)
            person_detected = True
            print(f"Person detected (stable): pin ON")
        elif not stable_person and person_detected:
            requests.get(f"{ESP32_BASE_URL}/person-gone", timeout=3.0)
            person_detected = False
            print(f"No person (stable): pin OFF")
    except requests.RequestException as e:
        print("ESP32 request failed:", e)

def main():
    cv2.namedWindow("ESP32-CAM Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESP32-CAM Detection", 960, 720)

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        person_present = detect_person(frame)
        trigger_esp32(person_present)

        cv2.imshow("ESP32-CAM Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
        time.sleep(FRAME_DELAY)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
