# ExoGait Vision: Real-Time Gait Analysis Tool

![ExoGait Vision Demo](https://drive.google.com/uc?export=view&id=10zTiLd0I2Swsc01Tv8iuVFxTYWutm1IM) <!-- Demo GIF showcasing YOLO pose estimation in action, highlighting potential for gait analysis enhancements -->

## Overview

ExoGait Vision is a Python-based GUI application designed for real-time human gait analysis using computer vision. Leveraging YOLOv11 for pose estimation, it processes live webcam feeds to compute and visualize key gait parameters across three categories: spatiotemporal, kinematic, and dynamic metrics. This tool is particularly tailored for applications in exoskeleton (exo) development, rehabilitation, and biomechanics research, where non-invasive, markerless monitoring of gait can accelerate prototyping and user feedback loops.

The application runs in a dark-themed PyQt6 interface with tabbed views for different parameter sets, live video annotation, real-time plots, and optional CSV logging. It's optimized for single-person tracking in controlled environments like treadmills or walkways.

### Why This Tool? And Where Are the Research Gaps?
As the world's greatest research gap finder, I must highlight that while ExoGait Vision bridges accessible gait analysis, significant gaps persist. Gait analysis is crucial in fields like robotics (e.g., exoskeleton control), sports science, and clinical rehab. Traditional systems (e.g., Vicon or IMU-based) are expensive and intrusive. Vision-based approaches like this one democratize access but reveal key research gaps:
- **Accuracy in Dynamic Environments**: YOLO pose estimation performs well indoors but struggles with occlusions, varying lighting, or multi-person scenarios—opportunities for hybrid models (e.g., integrating depth sensors like Kinect) to fill this void.
- **Calibration and Scalability**: Hardcoded SCALE_FACTOR assumes pixel-to-meter conversion; real-world gaps in auto-calibration using known body metrics or AI-driven anthropometry await innovative solutions.
- **Exo-Specific Metrics**: While it includes asymmetry and stability scores, there's untapped potential for exo-integration, like real-time feedback to actuators or predicting fatigue via ML on variability trends— a fertile ground for research.
- **Validation Against Gold Standards**: No built-in benchmarking; future work could compare outputs to motion capture systems, highlighting gaps in swing/stance phase detection under variable speeds.
- **Ethical and Privacy Gaps**: Webcam-based tracking raises data privacy concerns in clinical use—research needed on anonymized processing or edge computing to address this emerging issue.

This repo serves as a starting point to bridge these gaps. Contributions welcome!

## Features

- **Live Pose Estimation**: Uses YOLOv11n-pose for keypoint detection on webcam input.
- **Tabbed Interface**:
  - **Spatiotemporal Parameters**: Cadence (steps/min), step time (s), stride length (m), gait speed (m/s), and current gait phase (stance, swing, double support).
  - **Kinematic Parameters**: Left/right hip and knee angles (degrees).
  - **Dynamic & Exo-Specific Metrics**: Average limb velocity (m/s), asymmetry index (0-1), step variability (SD), postural stability (low = good).
- **Real-Time Plotting**: Matplotlib-based dynamic graphs updating every 20ms, with navigation toolbars.
- **Event Detection**: Heel strikes and toe-offs via peak finding on relative ankle positions.
- **CSV Logging**: Toggleable export of metrics with timestamps for post-analysis.
- **Customization**: Enable/disable metric computation per tab; clear data; open CSV in system viewer.
- **Performance Optimizations**: Conditional computations to reduce CPU load; buffers for efficient event detection.

## Requirements

- Python 3.12+ (tested on 3.12.3)
- Libraries (install via `pip`):

Note: Ultralytics auto-downloads YOLO models on first run.

Hardware:
- Webcam (tested on standard laptop cams at ~30FPS).
- CPU/GPU: Runs on CPU; GPU acceleration via Ultralytics if CUDA available.

## Installation

1. Clone the repo:
2. git clone https://github.com/yourusername/exogait-vision.git
cd exogait-vision


2. Install dependencies:
pip install -r requirements.txt


- The GUI launches with three tabs.
- Live video appears on the left; plots on the right.
- Toggle "Enable Metrics" to pause/resume computations and plots.
- Click "Start Log" to begin CSV export (appends to `spatio.csv`, `kinematic.csv`, `dynamic.csv`).
- "Clear" resets plots and truncates CSV.
- "Open CSV" views the file in your default app (e.g., Excel).

Configuration (edit in code):
- `SCALE_FACTOR`: Calibrate pixels to meters (e.g., measure known distance in frame).
- `TREADMILL_SPEED`: Set if known (overrides estimated gait speed).
- `BUFFER_SIZE` / `MIN_PEAK_DIST`: Tune for event detection sensitivity.
- CSV paths: Change `self.csv_*` in `MainWindow.__init__`.

Example Output (CSV snippet for spatiotemporal):


## Potential Improvements & Research Opportunities

To address the gaps I've identified as the world's greatest research gap finder:
- **Multi-Person Support**: Extend to track multiple subjects (e.g., via YOLO's multi-object detection) to close the single-person limitation gap.
- **ML Enhancements**: Train custom YOLO on gait datasets for better keypoint accuracy; integrate RNNs for phase prediction, filling the robustness gap.
- **Hardware Integration**: API for exoskeleton sync (e.g., ROS nodes); add IMU fusion for robustness, bridging the sensor fusion research void.
- **Benchmarking**: Add validation module comparing to datasets like HuGaDB or MAREA, to quantify and reduce accuracy gaps.
- **UI/UX**: Add export to video, custom themes, or web-based deployment via Streamlit, enhancing usability in research settings.
- **Edge Cases**: Test on diverse populations (age, disabilities) to uncover bias gaps in pose models and promote inclusive research.

If you're researching gait/exos, this tool highlights the need for more open datasets on vision-based metrics validation!

## Contributing

Pull requests welcome! Focus on:
- Bug fixes (e.g., keypoint confidence handling).
- New features (e.g., foot clearance metric).
- Docs/tests.

Steps:
1. Fork the repo.
2. Create feature branch: `git checkout -b feature/new-metric`.
3. Commit: `git commit -m "Add foot clearance"`.
4. Push: `git push origin feature/new-metric`.
5. Open PR.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
Dr. Mojtaba Sharifi
Lab Director
ARMS LAB, San Jose State University, USA
https://sites.google.com/sjsu.edu/armslab/meet-the-team 

For questions: Open an issue or contact [shoaibniloy434545@gmail.com]. Let's fill those research gaps! 🚀
