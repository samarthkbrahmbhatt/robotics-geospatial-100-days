# Robotics & Geospatial: 100 Days

One small build every day, alternating between **geospatial AI / remote sensing** and **robotics**. Each day lives in its own folder with code, results and notes.

## Progress

| Day | Project | Domain | Key skills |
|-----|---------|--------|------------|
| 01 | [NDVI over Abu Dhabi Eastern Mangroves](day-01-ndvi-mangroves) | Geospatial | Sentinel-2, STAC, NDVI, cloud masking |
| 02 | [2-link arm forward & inverse kinematics](day-02-arm-kinematics) | Robotics | FK, IK, law of cosines, matplotlib animation |
| 03 | [NDWI / MNDWI water mapping, Dubai coast](day-03-ndwi-water) | Geospatial | Water indices, Otsu thresholding, tile mosaicking |
| 04 | [PID control of a mass-damper system](day-04-pid-control) | Robotics | PID tuning, anti-windup, step response metrics |
| 05 | [EuroSAT land cover with a Random Forest](day-05-landcover-rf) | Geospatial + ML | Feature engineering, train/test split, confusion matrix |
| 06 | [A* path planning on a grid map](day-06-astar) | Robotics | A*, Dijkstra, admissible heuristics, priority queues |
| 07 | [Mangrove change detection, 2018 to 2024](day-07-mangrove-change) | Geospatial | Median compositing, change detection, seasonal matching |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
