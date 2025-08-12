# Traffic Flow Analysis

## Objective
Analyze traffic flow by counting vehicles in 3 lanes using YOLOv8 and tracking.

## Features
- Vehicle detection using YOLOv8 (pre-trained COCO model)
- ByteTrack-based tracking to prevent duplicate counts
- Lane-based counting for 3 distinct lanes
- Real-time visualization with lane boundaries and counts
- CSV export with vehicle ID, lane number, frame count, and timestamp

## Requirements
- Python 3.8+
- Packages in `requirements.txt`

## Installation
```bash
git clone <repo_url>
cd Traffic-Flow-Analysis
pip install -r requirements.txt
