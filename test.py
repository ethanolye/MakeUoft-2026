import cv2
import numpy as np
import urllib.request
import urllib.error
import socket
import requests
import time
import signal
import sys
import gc
import threading
import tkinter as tk
from tkinter import Label, Button, Frame
from pathlib import Path
from ultralytics import YOLO

# Set socket timeout globally (reduced for moving car)
socket.setdefaulttimeout(1.0)

# URL of your ESP32-CAM low-res endpoint (faster processing)
ESP32_CAM_URL = "http://172.20.10.2/cam-hijpg"  # Using LOW-RES for faster processing on moving car
ESP32_BASE_URL = "http://172.20.10.2"

# Pin configuration for detection states
# PIN1 & PIN3 are PWM pins (0-255 for motor speed)
# PIN2 & PIN4 are digital pins (0=LOW, 1=HIGH for motor direction)
PIN_CONFIG = {
    "person_detected": {  # When person is detected
        "pin1": 0,    # Motor 1 speed (0 = stopped)
        "pin2": 0,    # Motor 1 direction (LOW)
        "pin3": 0,    # Motor 2 speed (0 = stopped)
        "pin4": 0     # Motor 2 direction (LOW)
    },
    "person_gone": {      # When no person detected
        "pin1": 255,  # Motor 1 speed (255 = full speed)
        "pin2": 0,    # Motor 1 direction (LOW)
        "pin3": 255,  # Motor 2 speed (255 = full speed)
        "pin4": 0     # Motor 2 direction (LOW)
    },
    "forward_full": {     # Forward at full speed
        "pin1": 255,
        "pin2": 0,
        "pin3": 255,
        "pin4": 0
    },
    "backward_full": {    # Backward at full speed
        "pin1": 255,
        "pin2": 1,
        "pin3": 255,
        "pin4": 1
    },
    "forward_half": {     # Forward at half speed
        "pin1": 128,
        "pin2": 0,
        "pin3": 128,
        "pin4": 0
    },
    "backward_half": {    # Backward at half speed
        "pin1": 128,
        "pin2": 1,
        "pin3": 128,
        "pin4": 1
    },
    "turn_left": {        # Turn left (right motor forward, left motor stopped)
        "pin1": 0,
        "pin2": 0,
        "pin3": 200,
        "pin4": 0
    },
    "turn_right": {       # Turn right (left motor forward, right motor stopped)
        "pin1": 200,
        "pin2": 0,
        "pin3": 0,
        "pin4": 0
    },
    "stop": {             # Full stop
        "pin1": 0,
        "pin2": 0,
        "pin3": 0,
        "pin4": 0
    }
}

# Detection settings - OPTIMIZED FOR MOVING CAR
CONF_THRESHOLD = 0.45  # Slightly higher for more reliable detection on moving vehicle
FRAME_DELAY = 0.02     # ~50 FPS for faster response
DETECTION_BUFFER = 1   # Reduced to 1 for immediate response (fast-moving car)
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
running = True  # Global flag for graceful shutdown
frame_count = 0  # Track frames processed
manual_override = False  # Manual control flag
manual_motors_on = False  # Manual motor state
detection_running = True  # Flag for detection thread
status_label = None  # GUI status label reference

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n[INFO] Shutdown signal received. Exiting gracefully...")
    running = False

def manual_motors_on_click():
    """Manual override to turn motors ON."""
    global manual_override, manual_motors_on
    manual_override = True
    manual_motors_on = True
    try:
        # Use person_gone configuration for motors on
        config = PIN_CONFIG["person_gone"]
        params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
        requests.get(f"{ESP32_BASE_URL}/person-gone{params}", timeout=1.5)
        print(f"[MANUAL] Motors ON - Pins: {config}")
        if status_label:
            status_label.config(text="Status: MANUAL - Motors ON", fg="green")
    except Exception as e:
        print(f"[ERROR] Failed to turn motors on: {e}")

def manual_motors_off_click():
    """Manual override to turn motors OFF."""
    global manual_override, manual_motors_on
    manual_override = True
    manual_motors_on = False
    try:
        # Use person_detected configuration for motors off
        config = PIN_CONFIG["person_detected"]
        params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
        requests.get(f"{ESP32_BASE_URL}/person-detected{params}", timeout=1.5)
        print(f"[MANUAL] Motors OFF - Pins: {config}")
        if status_label:
            status_label.config(text="Status: MANUAL - Motors OFF", fg="red")
    except Exception as e:
        print(f"[ERROR] Failed to turn motors off: {e}")

def auto_mode_click():
    """Return to automatic detection mode."""
    global manual_override
    manual_override = False
    print("[MANUAL] Switched to AUTO mode")
    if status_label:
        status_label.config(text="Status: AUTO Detection Active", fg="blue")

def send_pin_config(config_name):
    """Send a specific pin configuration to ESP32."""
    global manual_override
    manual_override = True
    try:
        config = PIN_CONFIG[config_name]
        params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
        requests.get(f"{ESP32_BASE_URL}/person-gone{params}", timeout=1.5)
        print(f"[MANUAL] Config '{config_name}' - Pins: {config}")
        if status_label:
            status_label.config(text=f"Status: MANUAL - {config_name.replace('_', ' ').title()}", fg="orange")
    except Exception as e:
        print(f"[ERROR] Failed to send config '{config_name}': {e}")

def forward_full_click():
    send_pin_config("forward_full")

def backward_full_click():
    send_pin_config("backward_full")

def forward_half_click():
    send_pin_config("forward_half")

def backward_half_click():
    send_pin_config("backward_half")

def turn_left_click():
    send_pin_config("turn_left")

def turn_right_click():
    send_pin_config("turn_right")

def stop_click():
    send_pin_config("stop")

def create_control_gui(root):
    """Create manual control GUI window."""
    global status_label
    
    root.title("ESP32-CAM Motor Control")
    root.geometry("400x550")
    root.resizable(False, False)
    
    # Title label
    title_label = Label(root, text="Manual Motor Control", font=("Arial", 14, "bold"))
    title_label.pack(pady=5)
    
    # Status label
    status_label = Label(root, text="Status: AUTO Detection Active", font=("Arial", 10), fg="blue")
    status_label.pack(pady=5)
    
    # Main button frame
    button_frame = Frame(root)
    button_frame.pack(pady=5)
    
    # Basic controls section
    basic_label = Label(button_frame, text="Basic Controls", font=("Arial", 11, "bold"))
    basic_label.grid(row=0, column=0, columnspan=2, pady=5)
    
    motors_on_btn = Button(button_frame, text="Motors ON", command=manual_motors_on_click, 
                           width=18, bg="green", fg="white", font=("Arial", 9, "bold"))
    motors_on_btn.grid(row=1, column=0, padx=5, pady=3)
    
    motors_off_btn = Button(button_frame, text="Motors OFF", command=manual_motors_off_click, 
                            width=18, bg="red", fg="white", font=("Arial", 9, "bold"))
    motors_off_btn.grid(row=1, column=1, padx=5, pady=3)
    
    stop_btn = Button(button_frame, text="STOP", command=stop_click, 
                      width=18, bg="#8B0000", fg="white", font=("Arial", 9, "bold"))
    stop_btn.grid(row=2, column=0, padx=5, pady=3)
    
    auto_btn = Button(button_frame, text="AUTO Mode", command=auto_mode_click, 
                      width=18, bg="blue", fg="white", font=("Arial", 9, "bold"))
    auto_btn.grid(row=2, column=1, padx=5, pady=3)
    
    # Separator
    sep1 = Label(button_frame, text="" + u'\u2500'*40, font=("Arial", 8))
    sep1.grid(row=3, column=0, columnspan=2, pady=5)
    
    # Full speed section
    full_label = Label(button_frame, text="Full Speed", font=("Arial", 11, "bold"))
    full_label.grid(row=4, column=0, columnspan=2, pady=5)
    
    forward_full_btn = Button(button_frame, text="Forward Full", command=forward_full_click, 
                              width=18, bg="#006400", fg="white", font=("Arial", 9))
    forward_full_btn.grid(row=5, column=0, padx=5, pady=3)
    
    backward_full_btn = Button(button_frame, text="Backward Full", command=backward_full_click, 
                               width=18, bg="#8B4513", fg="white", font=("Arial", 9))
    backward_full_btn.grid(row=5, column=1, padx=5, pady=3)
    
    # Separator
    sep2 = Label(button_frame, text="" + u'\u2500'*40, font=("Arial", 8))
    sep2.grid(row=6, column=0, columnspan=2, pady=5)
    
    # Half speed section
    half_label = Label(button_frame, text="Half Speed", font=("Arial", 11, "bold"))
    half_label.grid(row=7, column=0, columnspan=2, pady=5)
    
    forward_half_btn = Button(button_frame, text="Forward Half", command=forward_half_click, 
                              width=18, bg="#228B22", fg="white", font=("Arial", 9))
    forward_half_btn.grid(row=8, column=0, padx=5, pady=3)
    
    backward_half_btn = Button(button_frame, text="Backward Half", command=backward_half_click, 
                               width=18, bg="#A0522D", fg="white", font=("Arial", 9))
    backward_half_btn.grid(row=8, column=1, padx=5, pady=3)
    
    # Separator
    sep3 = Label(button_frame, text="" + u'\u2500'*40, font=("Arial", 8))
    sep3.grid(row=9, column=0, columnspan=2, pady=5)
    
    # Turn section
    turn_label = Label(button_frame, text="Turning", font=("Arial", 11, "bold"))
    turn_label.grid(row=10, column=0, columnspan=2, pady=5)
    
    turn_left_btn = Button(button_frame, text="Turn Left", command=turn_left_click, 
                           width=18, bg="#4169E1", fg="white", font=("Arial", 9))
    turn_left_btn.grid(row=11, column=0, padx=5, pady=3)
    
    turn_right_btn = Button(button_frame, text="Turn Right", command=turn_right_click, 
                            width=18, bg="#4169E1", fg="white", font=("Arial", 9))
    turn_right_btn.grid(row=11, column=1, padx=5, pady=3)
    
    # Info label
    info_label = Label(root, text="Use buttons for manual control or switch to AUTO", 
                       font=("Arial", 8), fg="gray")
    info_label.pack(pady=5)

def detection_thread_worker():
    """Run detection loop in background thread."""
    global detection_running
    
    print("="*60)
    print("ESP32-CAM Person Detection System - WITH MANUAL CONTROL")
    print("="*60)
    print(f"ESP32 Camera URL: {ESP32_CAM_URL}")
    print(f"Confidence Threshold: {CONF_THRESHOLD}")
    print(f"Detection Buffer: {DETECTION_BUFFER} frames")
    print("="*60)
    
    # Create display window
    cv2.namedWindow("ESP32-CAM Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESP32-CAM Detection", 960, 720)
    
    consecutive_failures = 0
    max_failures = 5
    
    while detection_running:
        frame = get_frame()
        if frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                print(f"[ERROR] {consecutive_failures} consecutive failures. Reconnecting...")
                time.sleep(2)
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue
        
        if consecutive_failures > 0:
            print(f"[INFO] Connection restored after {consecutive_failures} failures")
            consecutive_failures = 0

        person_present = detect_person(frame)
        trigger_esp32(person_present)
        
        # Display frame
        cv2.imshow("ESP32-CAM Detection", frame)
        
        # Clean up frame memory
        del frame
        gc.collect()
        
        # Check for ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            print("[INFO] ESC key pressed. Exiting...")
            detection_running = False
            break
        
        time.sleep(FRAME_DELAY)
    
    cv2.destroyAllWindows()
    print("[INFO] Detection thread finished. Processed {} frames.".format(frame_count))

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    print("\n[INFO] Shutdown signal received. Exiting gracefully...")
    running = False

def get_frame():
    """Capture a single frame from ESP32-CAM with timeout protection."""
    try:
        # Use shorter timeout (1.0s) for faster failure detection on moving car
        img_resp = urllib.request.urlopen(ESP32_CAM_URL, timeout=1.5)
        img_data = img_resp.read()
        img_resp.close()
        
        # Process frame
        img_np = np.array(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Clean up numpy array
        del img_np
        del img_data
        
        return frame
    except (urllib.error.URLError, socket.timeout, Exception) as e:
        print(f"[WARNING] Failed to get frame: {type(e).__name__}")
        return None

def detect_person(frame):
    """Return True if person is detected above confidence threshold."""
    global max_confidence, frame_count
    
    # Run YOLOv8 inference
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    
    max_conf = 0.0
    person_found = False
    detected_persons = []
    
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
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        detected_persons.append({
                            'confidence': confidence,
                            'bbox': (x1, y1, x2, y2)
                        })
                        # Draw bounding box
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
    
    # Show frame count
    cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Log detections periodically (every 30 frames ~1 second)
    frame_count += 1
    if frame_count % 30 == 0:
        if person_found:
            print(f"[INFO] Frame {frame_count}: Person detected | Confidence: {max_conf:.2f} | Count: {len(detected_persons)}")
        else:
            print(f"[INFO] Frame {frame_count}: No person detected | Best: {max_conf:.2f}")
    
    max_confidence = max_conf
    return person_found

def trigger_esp32(person_present):
    """Call ESP32 endpoints to toggle detection pin with debouncing."""
    global person_detected, detection_history, manual_override
    
    # Skip auto-detection if in manual override mode
    if manual_override:
        return
    
    # Add to history
    detection_history.append(person_present)
    
    # Keep only recent history
    if len(detection_history) > DETECTION_BUFFER:
        detection_history.pop(0)
    
    # Only trigger if we have a stable detection
    stable_person = sum(detection_history) >= (DETECTION_BUFFER * 0.6)  # 60% consensus
    
    try:
        if stable_person and not person_detected:
            # Person detected - send configured pin states
            config = PIN_CONFIG["person_detected"]
            params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
            requests.get(f"{ESP32_BASE_URL}/person-detected{params}", timeout=1.5)
            person_detected = True
            print(f"[ACTION] Person detected (stable): Pins set to {config}")
        elif not stable_person and person_detected:
            # Person gone - send configured pin states
            config = PIN_CONFIG["person_gone"]
            params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
            requests.get(f"{ESP32_BASE_URL}/person-gone{params}", timeout=1.5)
            person_detected = False
            print(f"[ACTION] No person (stable): Pins set to {config}")
    except requests.RequestException as e:
        print(f"[ERROR] ESP32 request failed: {e}")

def main():
    global detection_running
    
    # Create root window (MUST be on main thread for Windows)
    root = tk.Tk()
    create_control_gui(root)
    
    # Start detection in background thread
    detection_thread = threading.Thread(target=detection_thread_worker, daemon=True)
    detection_thread.start()
    
    # Handle window close button
    def on_window_close():
        global detection_running
        detection_running = False
        root.destroy()
        time.sleep(0.5)
        sys.exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Run GUI mainloop (blocks until window closes)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_window_close()

if __name__ == "__main__":
    main()
