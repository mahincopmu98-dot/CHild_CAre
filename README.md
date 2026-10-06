# ChildCare AI

A webcam-based child safety demo that tracks people with YOLO, lets you mark a
child and caregiver, and displays alerts when the child enters a marked danger
zone or is too far from the caregiver.

## Requirements

- Python 3.9 or newer
- A webcam

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

From this directory, run:

```bash
python app.py
```

The app loads the included `yolov8n.pt` model and opens the default webcam.

## Controls

- Left-click to add points to the danger zone (three or more points make a
  polygon).
- Ctrl+left-click a detected person to assign the child.
- Right-click a detected person to assign the caregiver.
- Press `c` to clear the danger-zone points.
- Press `q` to quit.
