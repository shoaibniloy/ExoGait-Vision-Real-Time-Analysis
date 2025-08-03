import sys
import csv
import os
import subprocess
import platform
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QTabWidget, QGridLayout, QCheckBox, QPushButton, QHBoxLayout
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QColor, QPalette
import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
import time
from scipy.signal import find_peaks
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# Global histories for plots (time, value)
cadence_history = []
step_time_history = []
stride_length_history = []
gait_speed_history = []
left_hip_angle_history = []
left_knee_angle_history = []
right_hip_angle_history = []
right_knee_angle_history = []
limb_velocity_history = []
asymmetry_index_history = []
variability_index_history = []
stability_score_history = []

# Constants
BUFFER_SIZE = 100  # frames for event detection
MIN_PEAK_DIST = 15  # min frames between peaks (~0.5s at 30FPS)
SCALE_FACTOR = 1.0  # pixels to meters; calibrate e.g., via known leg length
TREADMILL_SPEED = 0.0  # m/s; set if known, else estimate from motion

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class PlotUpdater:
    def __init__(self, canvas, data_list, ylabel, color='b-'):
        self.canvas = canvas
        self.data_list = data_list
        self.ylabel = ylabel
        self.color = color
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(20)  # Update every 20ms
        self.paused = False

    def update_plot(self):
        if self.paused or not self.data_list:
            return
        times, values = zip(*self.data_list[-50:])  # Last 50 points for visibility
        self.canvas.axes.cla()
        self.canvas.axes.plot(times, values, self.color)
        self.canvas.axes.set_title(self.ylabel)
        self.canvas.axes.set_xlabel('Time (s)')
        self.canvas.axes.set_ylabel(self.ylabel)
        self.canvas.draw()

    def toggle_pause(self, paused):
        self.paused = paused

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    phase_signal = pyqtSignal(str)  # For current phase display

    def __init__(self):
        super().__init__()
        self.model = YOLO('yolo11n-pose.pt')
        self.running = True
        self.start_time = time.time()
        self.main = None  # Will be set to MainWindow instance
        # Buffers for relative positions (time, rel_x, rel_y)
        self.left_buffer = deque(maxlen=BUFFER_SIZE)
        self.right_buffer = deque(maxlen=BUFFER_SIZE)
        # Events: (time, leg, step_length at event)
        self.heel_strikes = deque(maxlen=50)  # Last 50 events for variability
        self.toe_offs = deque(maxlen=50)
        self.last_cadence = 0
        self.last_step_time = 0
        self.last_stride_length = 0
        self.last_gait_speed = 0
        self.current_phase = "Unknown"
        self.last_limb_velocity = 0
        self.last_asymmetry_index = 0
        self.last_variability_index = 0
        self.last_stability_score = 0
        self.last_left_hip = 0
        self.last_left_knee = 0
        self.last_right_hip = 0
        self.last_right_knee = 0
        self.com_buffer = deque(maxlen=BUFFER_SIZE)  # For center of mass stability

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                results = self.model(frame, verbose=False)
                annotated_frame = results[0].plot()
                curr_time = time.time() - self.start_time
                if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    kps = results[0].keypoints.xy[0].cpu().numpy()  # Assume first person
                    if len(kps) >= 17:
                        l_hip, r_hip = kps[11], kps[12]
                        mid_hip = (l_hip + r_hip) / 2 if np.all(l_hip > 0) and np.all(r_hip > 0) else (l_hip if np.all(l_hip > 0) else r_hip)
                        l_ankle, r_ankle = kps[15], kps[16]
                        # Append buffers always if available (lightweight)
                        if np.all(l_ankle > 0):
                            rel_l_x = (l_ankle[0] - mid_hip[0]) * SCALE_FACTOR
                            rel_l_y = (l_ankle[1] - mid_hip[1]) * SCALE_FACTOR
                            self.left_buffer.append((curr_time, rel_l_x, rel_l_y))
                        if np.all(r_ankle > 0):
                            rel_r_x = (r_ankle[0] - mid_hip[0]) * SCALE_FACTOR
                            rel_r_y = (r_ankle[1] - mid_hip[1]) * SCALE_FACTOR
                            self.right_buffer.append((curr_time, rel_r_x, rel_r_y))
                        if self.main.enable_dynamic and np.all(mid_hip > 0):
                            self.com_buffer.append((curr_time, mid_hip[1] * SCALE_FACTOR))
                        # Detect events only if needed
                        if self.main.enable_spatio or self.main.enable_dynamic:
                            self.detect_events()
                        # Compute angles only if needed
                        if self.main.enable_kinematic or self.main.enable_dynamic:
                            vec_vertical = np.array([0, 1.0])
                            # Left hip and knee
                            if all(np.all(kp > 0) for kp in [kps[11], kps[13], kps[15]]):
                                vec_thigh_l = kps[13] - kps[11]
                                norm_thigh_l = vec_thigh_l / np.linalg.norm(vec_thigh_l)
                                self.last_left_hip = np.degrees(np.arccos(np.clip(np.dot(norm_thigh_l, vec_vertical), -1.0, 1.0)))
                                left_hip_angle_history.append((curr_time, self.last_left_hip))
                                vec_to_hip_l = kps[11] - kps[13]
                                norm_to_hip_l = vec_to_hip_l / np.linalg.norm(vec_to_hip_l)
                                vec_to_ankle_l = kps[15] - kps[13]
                                norm_to_ankle_l = vec_to_ankle_l / np.linalg.norm(vec_to_ankle_l)
                                self.last_left_knee = np.degrees(np.arccos(np.clip(np.dot(norm_to_hip_l, norm_to_ankle_l), -1.0, 1.0)))
                                left_knee_angle_history.append((curr_time, self.last_left_knee))
                            # Right hip and knee
                            if all(np.all(kp > 0) for kp in [kps[12], kps[14], kps[16]]):
                                vec_thigh_r = kps[14] - kps[12]
                                norm_thigh_r = vec_thigh_r / np.linalg.norm(vec_thigh_r)
                                self.last_right_hip = np.degrees(np.arccos(np.clip(np.dot(norm_thigh_r, vec_vertical), -1.0, 1.0)))
                                right_hip_angle_history.append((curr_time, self.last_right_hip))
                                vec_to_hip_r = kps[12] - kps[14]
                                norm_to_hip_r = vec_to_hip_r / np.linalg.norm(vec_to_hip_r)
                                vec_to_ankle_r = kps[16] - kps[14]
                                norm_to_ankle_r = vec_to_ankle_r / np.linalg.norm(vec_to_ankle_r)
                                self.last_right_knee = np.degrees(np.arccos(np.clip(np.dot(norm_to_hip_r, norm_to_ankle_r), -1.0, 1.0)))
                                right_knee_angle_history.append((curr_time, self.last_right_knee))
                        # Compute params (conditional inside)
                        self.compute_params(curr_time)
                # Emit frame always
                rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(qt_image.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio))
                self.phase_signal.emit(self.current_phase)
            time.sleep(0.01)  # Throttle if too fast
        cap.release()

    def detect_events(self):
        for leg, buffer in [('left', self.left_buffer), ('right', self.right_buffer)]:
            if len(buffer) > MIN_PEAK_DIST:
                times, rel_xs, _ = zip(*buffer)
                rel_xs = np.array(rel_xs)
                # Heel strike: local max rel_x
                peaks, _ = find_peaks(rel_xs, distance=MIN_PEAK_DIST)
                for idx in peaks[-1:]:  # Only newest
                    event_time = list(buffer)[idx][0]
                    if not any(abs(event_time - t) < 0.1 for t, l, _ in self.heel_strikes if l == leg):
                        other_rel_x = list(self.right_buffer)[-1][1] if leg == 'left' else list(self.left_buffer)[-1][1]
                        step_length = abs(rel_xs[idx] - other_rel_x)
                        self.heel_strikes.append((event_time, leg, step_length))
                # Toe-off: local min rel_x
                valleys, _ = find_peaks(-rel_xs, distance=MIN_PEAK_DIST)
                for idx in valleys[-1:]:
                    event_time = list(buffer)[idx][0]
                    if not any(abs(event_time - t) < 0.1 for t, l in self.toe_offs if l == leg):
                        self.toe_offs.append((event_time, leg))

    def compute_params(self, curr_time):
        if self.main.enable_spatio:
            recent_strikes = [e for e in self.heel_strikes if curr_time - e[0] < 60]
            if recent_strikes:
                time_span = curr_time - min(t for t, _, _ in recent_strikes)
                num_steps = len(recent_strikes)
                self.last_cadence = num_steps / (time_span / 60) if time_span > 0 else 0
                cadence_history.append((curr_time, self.last_cadence))
                if num_steps > 2:
                    step_times = np.diff([t for t, _, _ in sorted(recent_strikes)])
                    self.last_variability_index = np.std(step_times[-10:])  # Shared with dynamic, but compute here
            sorted_strikes = sorted(self.heel_strikes)
            if len(sorted_strikes) > 1:
                step_times = [sorted_strikes[i+1][0] - sorted_strikes[i][0] for i in range(len(sorted_strikes)-1) if sorted_strikes[i][1] != sorted_strikes[i+1][1]]
                if step_times:
                    self.last_step_time = np.mean(step_times[-5:])
                    step_time_history.append((curr_time, self.last_step_time))
            recent_lengths = [l for _, _, l in recent_strikes if l > 0]
            if recent_lengths:
                self.last_stride_length = min(np.mean(recent_lengths) * 2, 1.2)
                stride_length_history.append((curr_time, self.last_stride_length))
            stride_time = self.last_step_time * 2 if self.last_step_time > 0 else 1
            self.last_gait_speed = TREADMILL_SPEED if TREADMILL_SPEED > 0 else (self.last_stride_length / stride_time if stride_time > 0 else 0)
            gait_speed_history.append((curr_time, self.last_gait_speed))
            last_hs = max((t for t, _, _ in self.heel_strikes), default=0)
            last_to = max((t for t, _ in self.toe_offs), default=0)
            if curr_time - last_hs < self.last_step_time * 0.6:
                self.current_phase = "Stance"
            elif curr_time - last_to < self.last_step_time * 0.4:
                self.current_phase = "Swing"
            else:
                self.current_phase = "Double Support"
        if self.main.enable_dynamic:
            if len(self.left_buffer) > 1 and len(self.right_buffer) > 1:
                l_times, l_xs, l_ys = zip(*list(self.left_buffer)[-2:])
                r_times, r_xs, r_ys = zip(*list(self.right_buffer)[-2:])
                dt_l = l_times[1] - l_times[0]
                dt_r = r_times[1] - r_times[0]
                if dt_l > 0 and dt_r > 0:
                    vel_l = np.sqrt((l_xs[1] - l_xs[0])**2 + (l_ys[1] - l_ys[0])**2) / dt_l
                    vel_r = np.sqrt((r_xs[1] - r_xs[0])**2 + (r_ys[1] - r_ys[0])**2) / dt_r
                    self.last_limb_velocity = (vel_l + vel_r) / 2
                    limb_velocity_history.append((curr_time, self.last_limb_velocity))
            if self.last_left_knee != 0 and self.last_right_knee != 0:
                self.last_asymmetry_index = abs(self.last_left_knee - self.last_right_knee) / max(self.last_left_knee, self.last_right_knee, 1)
                asymmetry_index_history.append((curr_time, self.last_asymmetry_index))
            recent_lengths = [l for _, _, l in [e for e in self.heel_strikes if curr_time - e[0] < 60] if l > 0]
            if len(recent_lengths) > 2:
                self.last_variability_index = np.std(recent_lengths[-10:])
                variability_index_history.append((curr_time, self.last_variability_index))
            recent_com = [y for t, y in self.com_buffer if curr_time - t < 10]
            if len(recent_com) > 2:
                self.last_stability_score = np.std(recent_com)
                stability_score_history.append((curr_time, self.last_stability_score))
        # Log to CSV if enabled (at end, using last_ values)
        if self.main.enable_spatio and self.main.logging_spatio:
            with open(self.main.csv_spatio, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([curr_time, self.last_cadence, self.last_step_time, self.last_stride_length, self.last_gait_speed])
        if self.main.enable_kinematic and self.main.logging_kinematic:
            with open(self.main.csv_kinematic, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([curr_time, self.last_left_hip, self.last_left_knee, self.last_right_hip, self.last_right_knee])
        if self.main.enable_dynamic and self.main.logging_dynamic:
            with open(self.main.csv_dynamic, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([curr_time, self.last_limb_velocity, self.last_asymmetry_index, self.last_variability_index, self.last_stability_score])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ExoGait Vision")
        self.setGeometry(100, 100, 1200, 800)
        # Enable flags and logging
        self.enable_spatio = True
        self.enable_kinematic = True
        self.enable_dynamic = True
        self.logging_spatio = False
        self.logging_kinematic = False
        self.logging_dynamic = False
        self.csv_spatio = 'spatio.csv'
        self.csv_kinematic = 'kinematic.csv'
        self.csv_dynamic = 'dynamic.csv'
        # Create tab widget
        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)
        # Setup tabs
        self.tab1 = QWidget()
        self.tab_widget.addTab(self.tab1, "Spatiotemporal Parameters")
        self.setup_tab(self.tab1, 'spatio')
        self.tab2 = QWidget()
        self.tab_widget.addTab(self.tab2, "Kinematic Parameters")
        self.setup_tab(self.tab2, 'kinematic')
        self.tab3 = QWidget()
        self.tab_widget.addTab(self.tab3, "Dynamic & Exo-Specific Metrics")
        self.setup_tab(self.tab3, 'dynamic')
        # Start video thread
        self.thread = VideoThread()
        self.thread.main = self
        self.thread.change_pixmap_signal.connect(self.update_image_tab1)
        self.thread.change_pixmap_signal.connect(self.update_image_tab2)
        self.thread.change_pixmap_signal.connect(self.update_image_tab3)
        self.thread.phase_signal.connect(self.update_phase)
        self.thread.start()

    def setup_tab(self, tab, tab_type):
        layout = QGridLayout()
        # Webcam feed (left side)
        video_label = QLabel(tab)
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(video_label, 0, 0, 12, 1)
        # Phase label only for spatio (below video)
        if tab_type == 'spatio':
            phase_label = QLabel("Current Phase: Unknown", tab)
            layout.addWidget(phase_label, 12, 0, 1, 1)
            self.phase_label = phase_label  # For update
            self.video_label1 = video_label
        elif tab_type == 'kinematic':
            self.video_label2 = video_label
        else:
            self.video_label3 = video_label
        # Plots (right side, 2x2 formation)
        plot_configs = {
            'spatio': [
                ('Cadence Plot', cadence_history, "Cadence (steps/min)", 'r-'),
                ('Step Time Plot', step_time_history, "Step Time (s)", 'g-'),
                ('Stride Length Plot', stride_length_history, "Stride Length (m)", 'y-'),
                ('Gait Speed Plot', gait_speed_history, "Gait Speed (m/s)", 'c-')
            ],
            'kinematic': [
                ('Left Hip Angle Plot', left_hip_angle_history, "Left Hip Angle (deg)", 'm-'),
                ('Left Knee Angle Plot', left_knee_angle_history, "Left Knee Angle (deg)", 'b-'),
                ('Right Hip Angle Plot', right_hip_angle_history, "Right Hip Angle (deg)", 'r-'),
                ('Right Knee Angle Plot', right_knee_angle_history, "Right Knee Angle (deg)", 'g-')
            ],
            'dynamic': [
                ('Average Limb Velocity Plot', limb_velocity_history, "Avg Limb Velocity (m/s)", 'y-'),
                ('Asymmetry Index Plot', asymmetry_index_history, "Asymmetry Index (0-1)", 'c-'),
                ('Step Variability Plot', variability_index_history, "Step Variability (SD)", 'm-'),
                ('Postural Stability Plot', stability_score_history, "Postural Stability (low=good)", 'b-')
            ]
        }[tab_type]
        updaters = []
        # Top left plot
        title = QLabel(plot_configs[0][0])
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 1)
        canvas = MplCanvas(tab, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(canvas, tab)
        layout.addWidget(toolbar, 1, 1)
        layout.addWidget(canvas, 2, 1, 4, 1)  # Span 4 rows for size
        updater = PlotUpdater(canvas, plot_configs[0][1], plot_configs[0][2], plot_configs[0][3])
        updaters.append(updater)
        # Top right plot
        title = QLabel(plot_configs[1][0])
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 0, 2)
        canvas = MplCanvas(tab, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(canvas, tab)
        layout.addWidget(toolbar, 1, 2)
        layout.addWidget(canvas, 2, 2, 4, 1)
        updater = PlotUpdater(canvas, plot_configs[1][1], plot_configs[1][2], plot_configs[1][3])
        updaters.append(updater)
        # Bottom left plot
        title = QLabel(plot_configs[2][0])
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 6, 1)
        canvas = MplCanvas(tab, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(canvas, tab)
        layout.addWidget(toolbar, 7, 1)
        layout.addWidget(canvas, 8, 1, 4, 1)
        updater = PlotUpdater(canvas, plot_configs[2][1], plot_configs[2][2], plot_configs[2][3])
        updaters.append(updater)
        # Bottom right plot
        title = QLabel(plot_configs[3][0])
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title, 6, 2)
        canvas = MplCanvas(tab, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(canvas, tab)
        layout.addWidget(toolbar, 7, 2)
        layout.addWidget(canvas, 8, 2, 4, 1)
        updater = PlotUpdater(canvas, plot_configs[3][1], plot_configs[3][2], plot_configs[3][3])
        updaters.append(updater)
        # Killswitch toggle and CSV buttons (horizontal below all)
        bottom_hbox = QHBoxLayout()
        toggle = QCheckBox("Enable Metrics")
        toggle.setChecked(True)
        toggle.setStyleSheet("QCheckBox { font-size: 20px; }")
        toggle.stateChanged.connect(lambda state: self.update_enable(tab_type, state == Qt.CheckState.Checked.value))
        toggle.stateChanged.connect(lambda state: [u.toggle_pause(state != Qt.CheckState.Checked.value) for u in updaters])
        bottom_hbox.addWidget(toggle)
        open_btn = QPushButton("Open CSV")
        open_btn.clicked.connect(lambda: self.open_csv(tab_type))
        bottom_hbox.addWidget(open_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.clear_data(tab_type))
        bottom_hbox.addWidget(clear_btn)
        start_stop_btn = QPushButton("Start Log")
        start_stop_btn.clicked.connect(lambda: self.toggle_logging(tab_type, start_stop_btn))
        bottom_hbox.addWidget(start_stop_btn)
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_hbox)
        layout.addWidget(bottom_widget, 12, 1, 1, 2)  # Below plots, span 2 columns
        tab.setLayout(layout)

    def update_enable(self, tab_type, enabled):
        if tab_type == 'spatio':
            self.enable_spatio = enabled
        elif tab_type == 'kinematic':
            self.enable_kinematic = enabled
        else:
            self.enable_dynamic = enabled

    def toggle_logging(self, tab_type, btn):
        if tab_type == 'spatio':
            self.logging_spatio = not self.logging_spatio
            btn.setText("Stop Log" if self.logging_spatio else "Start Log")
            if self.logging_spatio:
                header_needed = not os.path.exists(self.csv_spatio) or os.stat(self.csv_spatio).st_size == 0
                with open(self.csv_spatio, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if header_needed:
                        writer.writerow(['timestamp', 'cadence', 'step_time', 'stride_length', 'gait_speed'])
        elif tab_type == 'kinematic':
            self.logging_kinematic = not self.logging_kinematic
            btn.setText("Stop Log" if self.logging_kinematic else "Start Log")
            if self.logging_kinematic:
                header_needed = not os.path.exists(self.csv_kinematic) or os.stat(self.csv_kinematic).st_size == 0
                with open(self.csv_kinematic, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if header_needed:
                        writer.writerow(['timestamp', 'left_hip', 'left_knee', 'right_hip', 'right_knee'])
        else:
            self.logging_dynamic = not self.logging_dynamic
            btn.setText("Stop Log" if self.logging_dynamic else "Start Log")
            if self.logging_dynamic:
                header_needed = not os.path.exists(self.csv_dynamic) or os.stat(self.csv_dynamic).st_size == 0
                with open(self.csv_dynamic, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if header_needed:
                        writer.writerow(['timestamp', 'limb_velocity', 'asymmetry_index', 'variability_index', 'stability_score'])

    def clear_data(self, tab_type):
        if tab_type == 'spatio':
            cadence_history.clear()
            step_time_history.clear()
            stride_length_history.clear()
            gait_speed_history.clear()
            with open(self.csv_spatio, 'w', newline=''): pass  # Truncate
        elif tab_type == 'kinematic':
            left_hip_angle_history.clear()
            left_knee_angle_history.clear()
            right_hip_angle_history.clear()
            right_knee_angle_history.clear()
            with open(self.csv_kinematic, 'w', newline=''): pass
        else:
            limb_velocity_history.clear()
            asymmetry_index_history.clear()
            variability_index_history.clear()
            stability_score_history.clear()
            with open(self.csv_dynamic, 'w', newline=''): pass

    def open_csv(self, tab_type):
        file = getattr(self, f'csv_{tab_type}')
        system = platform.system()
        try:
            if system == 'Windows':
                os.startfile(file)
            elif system == 'Darwin':
                subprocess.call(['open', file])
            else:
                subprocess.call(['xdg-open', file])
        except Exception as e:
            print(f"Error opening CSV: {e}")

    def update_image_tab1(self, qt_image):
        self.video_label1.setPixmap(QPixmap.fromImage(qt_image))

    def update_image_tab2(self, qt_image):
        self.video_label2.setPixmap(QPixmap.fromImage(qt_image))

    def update_image_tab3(self, qt_image):
        self.video_label3.setPixmap(QPixmap.fromImage(qt_image))

    def update_phase(self, phase):
        self.phase_label.setText(f"Current Phase: {phase}")

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.wait()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Dark mode palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())