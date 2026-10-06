import cv2
import numpy as np
from ultralytics import YOLO

# YOLO মডেল লোড করা
model = YOLO('yolov8n.pt')

# গ্লোবাল ভ্যারিয়েবল
risk_zone_points = []
selected_child_id = None
selected_caregiver_id = None
tracked_persons = []

def mouse_callback(event, x, y, flags, param):
    global risk_zone_points, selected_child_id, selected_caregiver_id, tracked_persons
    
    # মাউসের বাম ক্লিক
    if event == cv2.EVENT_LBUTTONDOWN:
        # কীবোর্ডের Ctrl চেপে বাম ক্লিক করলে: CHILD সিলেক্ট হবে
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            for person in tracked_persons:
                px1, py1, px2, py2, pid = person
                if px1 <= x <= px2 and py1 <= y <= py2:
                    selected_child_id = pid
                    print(f"[INFO] Child Assigned to ID: {pid}")
                    break
        # সাধারণ বাম ক্লিক: Risk Zone-এর পয়েন্ট যোগ হবে
        else:
            risk_zone_points.append((x, y))
            print(f"[INFO] Risk Zone Point Added: ({x}, {y})")

    # মাউসের ডান ক্লিক: CAREGIVER সিলেক্ট হবে
    elif event == cv2.EVENT_RBUTTONDOWN:
        for person in tracked_persons:
            px1, py1, px2, py2, pid = person
            if px1 <= x <= px2 and py1 <= y <= py2:
                selected_caregiver_id = pid
                print(f"[INFO] Caregiver Assigned to ID: {pid}")
                break

# ল্যাপটপের ওয়েবক্যাম ওপেন করা
cap = cv2.VideoCapture(0)
cv2.namedWindow("ChildCare AI - Live Demo")
cv2.setMouseCallback("ChildCare AI - Live Demo", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ১. মানুষ ট্র্যাকিং
    results = model.track(frame, persist=True, classes=[0], verbose=False)
    tracked_persons = []

    child_center = None
    caregiver_center = None
    child_in_danger_zone = False

    # ২. Risk Zone আঁকা
    if len(risk_zone_points) >= 3:
        pts = np.array(risk_zone_points, np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "DANGER ZONE", (risk_zone_points[0][0], risk_zone_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif len(risk_zone_points) > 0:
        for pt in risk_zone_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    # ৩. মানুষকে বাউন্ডিং বক্সে দেখানো
    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            tracked_persons.append((x1, y1, x2, y2, track_id))

            center_x = int((x1 + x2) / 2)
            feet_y = y2
            center_y = int((y1 + y2) / 2)

            label = f"Person ID: {track_id}"
            box_color = (255, 255, 0)

            # রোল চিহ্নিত করা
            if track_id == selected_child_id:
                label = f"CHILD [ID:{track_id}]"
                box_color = (0, 255, 255)
                child_center = (center_x, center_y)

                # Risk Zone চেকিং
                if len(risk_zone_points) >= 3:
                    pts = np.array(risk_zone_points, np.int32)
                    dist = cv2.pointPolygonTest(pts, (center_x, feet_y), False)
                    if dist >= 0:
                        child_in_danger_zone = True

            elif track_id == selected_caregiver_id:
                label = f"CAREGIVER [ID:{track_id}]"
                box_color = (0, 255, 0)
                caregiver_center = (center_x, center_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # ৪. সেফটি অ্যালার্ট তৈরি
    alerts = []
    if child_in_danger_zone:
        alerts.append("CRITICAL: CHILD IN RISK ZONE!")

    if child_center:
        if caregiver_center:
            dist_px = np.linalg.norm(np.array(child_center) - np.array(caregiver_center))
            cv2.line(frame, child_center, caregiver_center, (255, 255, 255), 1)
            cv2.putText(frame, f"Dist: {int(dist_px)}px", 
                        (int((child_center[0] + caregiver_center[0])/2), int((child_center[1] + caregiver_center[1])/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if dist_px > 220:
                alerts.append("WARNING: CAREGIVER OUT OF SAFE DISTANCE (>2m)")
        else:
            alerts.append("WARNING: CHILD UNATTENDED! NO CAREGIVER NEARBY")

    # ৫. স্ক্রিনে লাল সতর্কবার্তা দেখানো
    if alerts:
        cv2.rectangle(frame, (10, 10), (620, 45 + len(alerts)*35), (0, 0, 180), -1)
        for idx, alert_msg in enumerate(alerts):
            cv2.putText(frame, alert_msg, (20, 40 + idx*35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "Click: Draw Zone | Ctrl+LClick: Set Child | RClick: Set Caregiver | 'c': Clear | 'q': Exit",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("ChildCare AI - Live Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        risk_zone_points = []

cap.release()
cv2.destroyAllWindows()