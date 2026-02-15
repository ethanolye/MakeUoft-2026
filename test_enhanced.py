"""
Enhanced ESP32-CAM Person Detection System
------------------------------------------
New Features:
1. Configuration file support (JSON)
2. Performance metrics & FPS counter
3. Enhanced logging with timestamps
4. Video recording capability
5. Statistics tracking & CSV logging
6. Keyboard shortcuts in display window
7. Detection zone configuration (ROI)
8. Connection health monitoring
9. Multi-resolution dynamic switching
10. Improved error handling & auto-reconnect
"""

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
from tkinter import Label, Button, Frame, Checkbutton, IntVar, Scale, HORIZONTAL
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import json
import csv
from collections import deque

# Set socket timeout globally
socket.setdefaulttimeout(1.0)

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "detection_config.json"
LOG_FILE = Path(__file__).parent / "detection_log.csv"

# Default configuration
DEFAULT_CONFIG = {
    "esp32_ip": "172.20.10.2",
    "camera_resolution": "hi",  # lo, mid, hi
    "conf_threshold": 0.45,
    "frame_delay": 0.02,
    "detection_buffer": 1,
    "enable_recording": False,
    "enable_csv_logging": True,
    "show_fps": True,
    "detection_zone": None,  # None or [x1, y1, x2, y2]
    "timeout": 1.0
}

# Load or create configuration
def load_config():
    """Load configuration from JSON file or create default."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                print(f"[CONFIG] Loaded from {CONFIG_FILE}")
                return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}. Using defaults.")
    
    # Save default config
    with open(CONFIG_FILE, 'w') as f:
        json.dump(DEFAULT_CONFIG, indent=4, fp=f)
        print(f"[CONFIG] Created default config at {CONFIG_FILE}")
    
    return DEFAULT_CONFIG.copy()

# Load configuration
config = load_config()

# URL of your ESP32-CAM endpoint
ESP32_BASE_URL = f"http://{config['esp32_ip']}"
ESP32_CAM_URL = f"{ESP32_BASE_URL}/cam-{config['camera_resolution']}.jpg"

# Pin configuration for detection states
PIN_CONFIG = {
    "person_detected": {
        "pin1": 0, "pin2": 255, "pin3": 0, "pin4": 255
    },
    "person_gone": {
        "pin1": 255, "pin2": 0, "pin3": 255, "pin4": 0
    },
    "forward_full": {
        "pin1": 255, "pin2": 0, "pin3": 255, "pin4": 0
    },
    "backward_full": {
        "pin1": 255, "pin2": 1, "pin3": 255, "pin4": 1
    },
    "forward_half": {
        "pin1": 128, "pin2": 0, "pin3": 128, "pin4": 0
    },
    "turn_left": {
        "pin1": 0, "pin2": 0, "pin3": 200, "pin4": 0
    },
    "turn_right": {
        "pin1": 200, "pin2": 0, "pin3": 0, "pin4": 0
    },
    "stop": {
        "pin1": 0, "pin2": 0, "pin3": 0, "pin4": 0
    }
}

# Detection settings
CONF_THRESHOLD = config['conf_threshold']
FRAME_DELAY = config['frame_delay']
DETECTION_BUFFER = config['detection_buffer']
TIMEOUT = config['timeout']

# Model setup
script_dir = Path(__file__).parent
model_path = script_dir / "yolov8n.pt"

if not model_path.exists():
    print(f"[ERROR] {model_path} not found.")
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"[MODEL] Loading YOLOv8 from {model_path}...")
model = YOLO(str(model_path))
print("[MODEL] Loaded successfully!")

# Global state variables
person_detected = False
detection_history = deque(maxlen=DETECTION_BUFFER)
max_confidence = 0.0
running = True
frame_count = 0
manual_override = False
detection_running = True
status_label = None
fps_label = None
stats_label = None

# Performance tracking
fps_queue = deque(maxlen=30)
last_frame_time = time.time()

# Statistics
stats = {
    "total_detections": 0,
    "total_frames": 0,
    "avg_confidence": 0.0,
    "connection_failures": 0,
    "start_time": datetime.now()
}

# Video recording
video_writer = None
recording_enabled = config['enable_recording']

# CSV logging
csv_logging_enabled = config['enable_csv_logging']
if csv_logging_enabled and not LOG_FILE.exists():
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Event', 'Confidence', 'Frame', 'FPS'])

def log_to_csv(event, confidence=0.0):
    """Log detection events to CSV file."""
    if not csv_logging_enabled:
        return
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            current_fps = calculate_fps()
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                event,
                f"{confidence:.3f}",
                frame_count,
                f"{current_fps:.1f}"
            ])
    except Exception as e:
        print(f"[ERROR] CSV logging failed: {e}")

def calculate_fps():
    """Calculate current FPS from the queue."""
    if len(fps_queue) < 2:
        return 0.0
    time_diffs = [fps_queue[i] - fps_queue[i-1] for i in range(1, len(fps_queue))]
    avg_time = sum(time_diffs) / len(time_diffs)
    return 1.0 / avg_time if avg_time > 0 else 0.0

def update_stats_display():
    """Update statistics label in GUI."""
    if stats_label:
        uptime = (datetime.now() - stats['start_time']).total_seconds()
        stats_text = (f"Frames: {stats['total_frames']} | "
                     f"Detections: {stats['total_detections']} | "
                     f"Fails: {stats['connection_failures']} | "
                     f"Uptime: {uptime:.0f}s")
        stats_label.config(text=stats_text)

def log_message(level, message):
    """Enhanced logging with timestamp and level."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level}] {message}")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global running
    log_message("INFO", "Shutdown signal received. Exiting...")
    running = False

def send_pin_config(config_name, show_status=True):
    """Send a specific pin configuration to ESP32."""
    global manual_override
    manual_override = True
    try:
        config = PIN_CONFIG[config_name]
        params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
        requests.get(f"{ESP32_BASE_URL}/person-gone{params}", timeout=TIMEOUT)
        if show_status:
            log_message("MANUAL", f"Config '{config_name}' - Pins: {config}")
            if status_label:
                status_label.config(text=f"Status: MANUAL - {config_name.replace('_', ' ').title()}", fg="orange")
    except Exception as e:
        log_message("ERROR", f"Failed to send config '{config_name}': {e}")

# Manual control functions
def manual_motors_on_click():
    send_pin_config("person_gone")
    if status_label:
        status_label.config(text="Status: MANUAL - Motors ON", fg="green")

def manual_motors_off_click():
    send_pin_config("person_detected")
    if status_label:
        status_label.config(text="Status: MANUAL - Motors OFF", fg="red")

def auto_mode_click():
    global manual_override
    manual_override = False
    log_message("MANUAL", "Switched to AUTO mode")
    if status_label:
        status_label.config(text="Status: AUTO Detection Active", fg="blue")

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

def toggle_recording():
    """Toggle video recording on/off."""
    global recording_enabled
    recording_enabled = not recording_enabled
    status = "ENABLED" if recording_enabled else "DISABLED"
    log_message("INFO", f"Recording {status}")

def change_resolution(res):
    """Change camera resolution dynamically."""
    global ESP32_CAM_URL
    ESP32_CAM_URL = f"{ESP32_BASE_URL}/cam-{res}.jpg"
    log_message("INFO", f"Switched to {res.upper()} resolution")

def create_control_gui(root):
    """Create enhanced control GUI window."""
    global status_label, fps_label, stats_label
    
    root.title("ESP32-CAM Enhanced Control")
    root.geometry("450x650")
    root.resizable(False, False)
    
    # Title
    title_label = Label(root, text="Enhanced Motor Control", font=("Arial", 14, "bold"))
    title_label.pack(pady=5)
    
    # Status label
    status_label = Label(root, text="Status: AUTO Detection Active", font=("Arial", 10), fg="blue")
    status_label.pack(pady=3)
    
    # FPS label
    fps_label = Label(root, text="FPS: 0.0", font=("Arial", 9), fg="gray")
    fps_label.pack(pady=2)
    
    # Stats label
    stats_label = Label(root, text="Stats: Initializing...", font=("Arial", 8), fg="gray")
    stats_label.pack(pady=2)
    
    # Main controls
    button_frame = Frame(root)
    button_frame.pack(pady=5)
    
    # Basic Controls
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
    Label(button_frame, text="─"*40, font=("Arial", 8)).grid(row=3, column=0, columnspan=2, pady=3)
    
    # Full Speed
    Label(button_frame, text="Full Speed", font=("Arial", 11, "bold")).grid(row=4, column=0, columnspan=2, pady=3)
    
    Button(button_frame, text="Forward Full", command=forward_full_click,
           width=18, bg="#006400", fg="white", font=("Arial", 9)).grid(row=5, column=0, padx=5, pady=3)
    
    Button(button_frame, text="Backward Full", command=backward_full_click,
           width=18, bg="#8B4513", fg="white", font=("Arial", 9)).grid(row=5, column=1, padx=5, pady=3)
    
    # Separator
    Label(button_frame, text="─"*40, font=("Arial", 8)).grid(row=6, column=0, columnspan=2, pady=3)
    
    # Half Speed
    Label(button_frame, text="Half Speed", font=("Arial", 11, "bold")).grid(row=7, column=0, columnspan=2, pady=3)
    
    Button(button_frame, text="Forward Half", command=forward_half_click,
           width=18, bg="#228B22", fg="white", font=("Arial", 9)).grid(row=8, column=0, padx=5, pady=3)
    
    Button(button_frame, text="Backward Half", command=backward_half_click,
           width=18, bg="#A0522D", fg="white", font=("Arial", 9)).grid(row=8, column=1, padx=5, pady=3)
    
    # Separator
    Label(button_frame, text="─"*40, font=("Arial", 8)).grid(row=9, column=0, columnspan=2, pady=3)
    
    # Turning
    Label(button_frame, text="Turning", font=("Arial", 11, "bold")).grid(row=10, column=0, columnspan=2, pady=3)
    
    Button(button_frame, text="Turn Left", command=turn_left_click,
           width=18, bg="#4169E1", fg="white", font=("Arial", 9)).grid(row=11, column=0, padx=5, pady=3)
    
    Button(button_frame, text="Turn Right", command=turn_right_click,
           width=18, bg="#4169E1", fg="white", font=("Arial", 9)).grid(row=11, column=1, padx=5, pady=3)
    
    # Separator
    Label(button_frame, text="─"*40, font=("Arial", 8)).grid(row=12, column=0, columnspan=2, pady=3)
    
    # Resolution Controls
    Label(button_frame, text="Camera Resolution", font=("Arial", 11, "bold")).grid(row=13, column=0, columnspan=2, pady=3)
    
    Button(button_frame, text="Low Res (Fast)", command=lambda: change_resolution("lo"),
           width=18, bg="#696969", fg="white", font=("Arial", 8)).grid(row=14, column=0, padx=5, pady=2)
    
    Button(button_frame, text="High Res (Slow)", command=lambda: change_resolution("hi"),
           width=18, bg="#696969", fg="white", font=("Arial", 8)).grid(row=14, column=1, padx=5, pady=2)
    
    # Info
    info_label = Label(root, text="Keyboard: R=Record, Q=Quit, Space=Toggle Auto",
                       font=("Arial", 8), fg="gray")
    info_label.pack(pady=5)
    
    # Update GUI periodically
    def update_gui():
        if fps_label:
            current_fps = calculate_fps()
            fps_label.config(text=f"FPS: {current_fps:.1f}")
        update_stats_display()
        root.after(500, update_gui)
    
    update_gui()

def get_frame():
    """Capture a single frame from ESP32-CAM with timeout protection."""
    try:
        img_resp = urllib.request.urlopen(ESP32_CAM_URL, timeout=TIMEOUT)
        img_data = img_resp.read()
        img_resp.close()
        
        img_np = np.array(bytearray(img_data), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        del img_np, img_data
        return frame
    except (urllib.error.URLError, socket.timeout, Exception) as e:
        stats['connection_failures'] += 1
        return None

def detect_person(frame):
    """Return True if person is detected above confidence threshold."""
    global max_confidence, frame_count, video_writer
    
    # Apply detection zone if configured
    detection_zone = config.get('detection_zone')
    if detection_zone:
        x1, y1, x2, y2 = detection_zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame
    
    # Run YOLOv8 inference
    results = model(roi, conf=CONF_THRESHOLD, verbose=False)
    
    max_conf = 0.0
    person_found = False
    detected_persons = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == 0:  # Person class
                    max_conf = max(max_conf, confidence)
                    if confidence > CONF_THRESHOLD:
                        person_found = True
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Adjust coordinates if using detection zone
                        if detection_zone:
                            x1 += detection_zone[0]
                            y1 += detection_zone[1]
                            x2 += detection_zone[0]
                            y2 += detection_zone[1]
                        
                        detected_persons.append({'confidence': confidence, 'bbox': (x1, y1, x2, y2)})
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {confidence:.2f}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw status overlay
    status_color = (0, 255, 0) if person_found else (0, 0, 255)
    status_text = f"PERSON DETECTED (conf: {max_conf:.2f})" if person_found else f"No person (best: {max_conf:.2f})"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Show additional info
    cv2.putText(frame, f"Buffer: {len(detection_history)}/{DETECTION_BUFFER}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    current_fps = calculate_fps()
    if config['show_fps']:
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Recording indicator
    if recording_enabled and video_writer is not None:
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (frame.shape[1] - 70, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Update statistics
    stats['total_frames'] += 1
    frame_count += 1
    
    if person_found:
        stats['total_detections'] += 1
        stats['avg_confidence'] = ((stats['avg_confidence'] * (stats['total_detections'] - 1) + max_conf) /
                                   stats['total_detections'])
    
    # Video recording
    if recording_enabled:
        if video_writer is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = script_dir / f"recording_{timestamp}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(str(video_filename), fourcc, 20.0,
                                          (frame.shape[1], frame.shape[0]))
            log_message("INFO", f"Started recording: {video_filename}")
        video_writer.write(frame)
    elif video_writer is not None:
        video_writer.release()
        video_writer = None
        log_message("INFO", "Stopped recording")
    
    max_confidence = max_conf
    return person_found

def trigger_esp32(person_present):
    """Call ESP32 endpoints with debouncing."""
    global person_detected, detection_history, manual_override
    
    if manual_override:
        return
    
    detection_history.append(person_present)
    
    # Stable detection check
    stable_person = sum(detection_history) >= max(1, int(DETECTION_BUFFER * 0.6))
    
    try:
        if stable_person and not person_detected:
            config = PIN_CONFIG["person_detected"]
            params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
            requests.get(f"{ESP32_BASE_URL}/person-detected{params}", timeout=TIMEOUT)
            person_detected = True
            log_message("ACTION", f"Person detected (stable): Pins set to {config}")
            log_to_csv("PERSON_DETECTED", max_confidence)
        elif not stable_person and person_detected:
            config = PIN_CONFIG["person_gone"]
            params = f"?pin1={config['pin1']}&pin2={config['pin2']}&pin3={config['pin3']}&pin4={config['pin4']}"
            requests.get(f"{ESP32_BASE_URL}/person-gone{params}", timeout=TIMEOUT)
            person_detected = False
            log_message("ACTION", f"No person (stable): Pins set to {config}")
            log_to_csv("PERSON_GONE", max_confidence)
    except requests.RequestException as e:
        log_message("ERROR", f"ESP32 request failed: {e}")

def detection_thread_worker():
    """Run detection loop in background thread."""
    global detection_running, last_frame_time
    
    log_message("INFO", "="*60)
    log_message("INFO", "Enhanced ESP32-CAM Person Detection System")
    log_message("INFO", "="*60)
    log_message("INFO", f"Camera URL: {ESP32_CAM_URL}")
    log_message("INFO", f"Confidence Threshold: {CONF_THRESHOLD}")
    log_message("INFO", f"Detection Buffer: {DETECTION_BUFFER} frames")
    log_message("INFO", f"Recording: {'ENABLED' if recording_enabled else 'DISABLED'}")
    log_message("INFO", f"CSV Logging: {'ENABLED' if csv_logging_enabled else 'DISABLED'}")
    log_message("INFO", "="*60)
    log_message("INFO", "Keyboard Shortcuts:")
    log_message("INFO", "  ESC/Q - Quit")
    log_message("INFO", "  R - Toggle Recording")
    log_message("INFO", "  SPACE - Toggle Auto/Manual")
    log_message("INFO", "  1/2/3 - Change Resolution (Low/Mid/High)")
    log_message("INFO", "="*60)
    
    cv2.namedWindow("ESP32-CAM Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESP32-CAM Detection", 960, 720)
    
    consecutive_failures = 0
    max_failures = 5
    
    while detection_running:
        frame = get_frame()
        
        if frame is None:
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                log_message("ERROR", f"{consecutive_failures} consecutive failures. Reconnecting...")
                time.sleep(2)
                consecutive_failures = 0
            time.sleep(0.05)
            continue
        
        if consecutive_failures > 0:
            log_message("INFO", f"Connection restored after {consecutive_failures} failures")
            consecutive_failures = 0
        
        # Track FPS
        current_time = time.time()
        fps_queue.append(current_time)
        last_frame_time = current_time
        
        # Process detection
        person_present = detect_person(frame)
        trigger_esp32(person_present)
        
        # Display frame
        cv2.imshow("ESP32-CAM Detection", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or Q
            log_message("INFO", "Quit key pressed. Exiting...")
            detection_running = False
            break
        elif key == ord('r'):  # Toggle recording
            toggle_recording()
        elif key == ord(' '):  # Toggle auto/manual
            auto_mode_click()
        elif key == ord('1'):  # Low res
            change_resolution("lo")
        elif key == ord('2'):  # Mid res
            change_resolution("mid")
        elif key == ord('3'):  # High res
            change_resolution("hi")
        
        # Cleanup
        del frame
        gc.collect()
        time.sleep(FRAME_DELAY)
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    log_message("INFO", f"Detection thread finished. Processed {frame_count} frames.")
    log_message("INFO", f"Total detections: {stats['total_detections']}")
    log_message("INFO", f"Average confidence: {stats['avg_confidence']:.3f}")

def main():
    global detection_running
    
    # Signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create GUI
    root = tk.Tk()
    create_control_gui(root)
    
    # Start detection thread
    detection_thread = threading.Thread(target=detection_thread_worker, daemon=True)
    detection_thread.start()
    
    # Handle window close
    def on_window_close():
        global detection_running
        detection_running = False
        root.destroy()
        time.sleep(0.5)
        sys.exit(0)
    
    root.protocol("WM_DELETE_WINDOW", on_window_close)
    
    # Run GUI
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_window_close()

if __name__ == "__main__":
    main()
