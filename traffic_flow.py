import cv2
import pandas as pd
import time
import os
from ultralytics import YOLO
import yt_dlp
import datetime

# -----------------------
# Step 1: Prepare folders
# -----------------------
os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)

video_path = "data/traffic.mp4"

# -----------------------
# Step 2: Download video if not exists
# -----------------------
if not os.path.exists(video_path):
    print("Downloading video...")
    url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    ydl_opts = {'outtmpl': video_path, 'format': 'best'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("Download complete.")

# -----------------------
# Step 3: Lane calibration
# -----------------------
lanes = []
drawing = False
start_point = None
current_lane = None

def mouse_draw(event, x, y, flags, param):
    global drawing, start_point, current_lane, lanes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_lane = (start_point[0], start_point[1], x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1, x2, y2 = start_point[0], start_point[1], x, y
        lanes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        print(f"Lane {len(lanes)}: {lanes[-1]}")

cap = cv2.VideoCapture(video_path)
ret, first_frame = cap.read()
if not ret:
    print("❌ Could not read video file")
    exit()

print("\n🎯 Draw lanes by dragging with mouse (top to bottom). Press ENTER when done.")
cv2.namedWindow("Calibrate Lanes")
cv2.setMouseCallback("Calibrate Lanes", mouse_draw)

while True:
    display = first_frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(lanes, 1):
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"Lane {i}", (x1+5, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if current_lane:
        cv2.rectangle(display, (current_lane[0], current_lane[1]),
                      (current_lane[2], current_lane[3]), (255, 0, 0), 2)
    cv2.imshow("Calibrate Lanes", display)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter
        lanes.sort(key=lambda l: l[1])
        print("✅ Lanes sorted (top to bottom).")
        break
    elif key == 27:  # ESC
        print("❌ Calibration cancelled")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# -----------------------
# Step 4: YOLO detection
# -----------------------
model = YOLO("yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]

counts = {i+1: 0 for i in range(len(lanes))}
counted_vehicles = set()
results_data = []
frame_count = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    results = model.track(frame, persist=True, classes=vehicle_classes, tracker="bytetrack.yaml")

    if results and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for vid, box in zip(ids, boxes):
            if vid in counted_vehicles:
                continue
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            for lane_id, (lx1, ly1, lx2, ly2) in enumerate(lanes, 1):
                if lx1 < cx < lx2 and ly1 < cy < ly2:
                    counts[lane_id] += 1
                    counted_vehicles.add(vid)
                    results_data.append([
                        int(vid),
                        lane_id,
                        frame_count,
                        datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")
                    ])
                    break

    for lane_id, (lx1, ly1, lx2, ly2) in enumerate(lanes, 1):
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        cv2.putText(frame, f"Lane {lane_id}: {counts[lane_id]}",
                    (lx1+5, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Traffic Flow Analysis", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------
# Step 5: Save CSV with unique name
# -----------------------
if results_data:
    df = pd.DataFrame(results_data, columns=["Vehicle ID", "Lane", "Frame", "Timestamp"])
    df.sort_values(by="Frame", inplace=True)
    
    unique_filename = f"output/vehicle_counts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(unique_filename, index=False)
    
    print(f"✅ CSV saved as {unique_filename} with {len(results_data)} rows")
else:
    print("⚠ No vehicles counted — check lanes.")

# -----------------------
# Step 6: Summary
# -----------------------
print("\nFinal Vehicle Counts per Lane:")
for lane_id in counts:
    print(f"Lane {lane_id}: {counts[lane_id]}")
