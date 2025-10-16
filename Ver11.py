#!/usr/bin/env python3
"""
Updated larva_tracker.py
updated from ver 9- in testing
An application for multi–animal larva tracking with robust motion modeling.
It uses a median background model, a Kalman filter for robust tracking,
bidirectional batch processing (forward and reverse), and a Hungarian assignment
to merge trajectories. A Tkinter GUI lets you select a video, start a live preview,
and stop tracking safely.

Key updates:
 - All Kalman filter matrices and measurements are now explicitly np.float32.
 - The tracker now uses a global data–association step (Hungarian algorithm) to
   match detections to predictions rather than iterating over identities one by one.
 - Lost larvae are not immediately removed. Instead, a missed–frame counter is used
   so that tracks are maintained for up to 50 frames without a detection before removal.
 - When a detection is re–acquired, we use the Kalman filter prediction along with
   other similarity measures (moment, heading, size, etc.) to reassign identities.
 - Improved feedback: console logging and GUI status updates to show progress.
 - Fixed drawing issues by converting coordinates to integers before calling OpenCV drawing routines.
"""

import numpy as np
import cv2
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import trackpy
from skimage.morphology import skeletonize
from skimage import img_as_bool
import time

# ----------------------------
# Preprocessing functions
# ----------------------------


def compute_background(video_path, num_frames=30):
    """
    Compute a median background from the first num_frames of the video.
    Returns a grayscale image.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1
    cap.release()
    if len(frames) == 0:
        return None
    median_bg = np.median(np.array(frames), axis=0).astype(np.uint8)
    print("Background computed.")
    return median_bg


def process_frame(frame, background, thresh_val=30):
    """
    Given a video frame and a background model, perform background subtraction,
    thresholding and basic morphological operations.
    Returns a binary mask.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)
    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def get_skeleton_endpoints(skel):
    """
    Given a skeletonized binary image (boolean array), return a list of endpoints.
    An endpoint is defined as a white pixel with only one white neighbor.
    """
    endpoints = []
    coords = np.column_stack(np.where(skel))
    for y, x in coords:
        y_min = max(y - 1, 0)
        y_max = min(y + 2, skel.shape[0])
        x_min = max(x - 1, 0)
        x_max = min(x + 2, skel.shape[1])
        neighborhood = skel[y_min:y_max, x_min:x_max]
        count = np.sum(neighborhood) - 1
        if count == 1:
            endpoints.append((x, y))
    return list(set(endpoints))


def extract_objects(mask, min_area=300, max_area=10000):
    """
    From a binary mask, find contours and—for contours whose area is in range—
    compute the centroid, skeletonize the candidate and extract its endpoints.
    Returns a list of tuples: (contour, centroid, skeleton, endpoints)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centroid = (cx, cy)
        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, -1)
        bool_mask = img_as_bool(obj_mask)
        skel = skeletonize(bool_mask)
        endpoints = get_skeleton_endpoints(skel)
        detected.append((cnt, centroid, skel, endpoints))
    return detected


# ----------------------------
# Robust motion model classes
# ----------------------------


@dataclass
class MotionState:
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    covariance: np.ndarray  # 4x4 state covariance
    direction_confidence: float  # Confidence in head/tail assignment


class RobustLarva:
    """
    Represents one larva with a robust (Kalman–filtered) state.
    Uses a constant–velocity model. A separate update method (update_state)
    is used to perform the Kalman correction with a new measurement.
    """

    def __init__(self, id: int, initial_position: Tuple[float, float]):
        self.id = id
        self.track_buffer = deque(maxlen=120)  # Store recent positions and confidence
        self.state = MotionState(
            position=np.array(initial_position, dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            covariance=np.eye(4, dtype=np.float32) * 100,
            direction_confidence=0.5,
        )
        self.kalman = self._initialize_kalman()
        self.head = None
        self.tail = None
        self.missed_frames = 0  # Counter for frames without an assigned detection

    def _initialize_kalman(self):
        """Initialize a constant–velocity Kalman filter with 32-bit floats."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.statePre = np.array(
            [[self.state.position[0]], [self.state.position[1]], [0], [0]], np.float32
        )
        return kf

    def predict(self):
        """
        Call the Kalman filter prediction.
        Returns the predicted position as a 2D point.
        """
        pred = self.kalman.predict()  # (4,1) array
        return pred[:2].flatten()

    def update_state(
        self, measurement: np.ndarray, endpoints: List[Tuple[float, float]]
    ):
        """
        Correct the Kalman filter with the new measurement.
        Update position, velocity, and head/tail assignments.
        """
        meas = np.array([[measurement[0]], [measurement[1]]], np.float32)
        corrected = self.kalman.correct(meas)
        self.state.position = corrected[:2].flatten()
        self.state.velocity = corrected[2:].flatten()
        # Update head/tail if motion is significant and endpoints exist.
        vel_magnitude = np.linalg.norm(self.state.velocity)
        if vel_magnitude > 1.0 and endpoints:
            velocity_direction = self.state.velocity / vel_magnitude
            best_dot = -np.inf
            head_candidate = None
            for pt in endpoints:
                vec = np.array(pt, np.float32) - self.state.position
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 0:
                    vec = vec / norm_vec
                    dot = np.dot(velocity_direction, vec)
                    if dot > best_dot:
                        best_dot = dot
                        head_candidate = pt
            self.head = head_candidate
            if len(endpoints) >= 2:
                tail_candidate = None
                max_dist = 0
                for pt in endpoints:
                    if pt == head_candidate:
                        continue
                    d = np.linalg.norm(
                        np.array(pt, np.float32) - np.array(head_candidate, np.float32)
                    )
                    if d > max_dist:
                        max_dist = d
                        tail_candidate = pt
                self.tail = tail_candidate
            else:
                self.tail = None
            # Smoothly update direction confidence.
            endpoint_scores = []
            for pt in endpoints:
                vec = np.array(pt, np.float32) - self.state.position
                norm_vec = np.linalg.norm(vec)
                if norm_vec > 0:
                    vec = vec / norm_vec
                    score = np.dot(velocity_direction, vec)
                    endpoint_scores.append(score)
            if endpoint_scores:
                max_score = max(endpoint_scores)
                self.state.direction_confidence = (
                    0.8 * self.state.direction_confidence
                    + 0.2 * (0.5 + 0.5 * max_score)
                )
        self.track_buffer.append(
            (self.state.position.copy(), self.state.direction_confidence)
        )
        self.missed_frames = 0  # Reset missed frame counter after a successful update


class RobustLarvaTracker:
    """
    Manages multiple RobustLarva objects by associating new detections
    (using a global Hungarian assignment based on Kalman predictions) and updating or
    creating larva objects. Lost larvae are kept for several frames (via missed_frames)
    to allow identity re–assignment.
    """

    def __init__(self, distance_threshold=50):
        self.larvae = {}  # id -> RobustLarva
        self.next_id = 0
        self.distance_threshold = distance_threshold

    def update(
        self,
        detected_objects: List[
            Tuple[np.ndarray, Tuple[int, int], np.ndarray, List[Tuple[int, int]]]
        ],
    ):
        # Prepare detection measurements and endpoints.
        detection_measurements = []
        detection_endpoints = []
        for det in detected_objects:
            _, centroid, _, endpoints = det
            detection_measurements.append(np.array(centroid, np.float32))
            detection_endpoints.append(endpoints)
        detection_measurements = np.array(
            detection_measurements
        )  # Expected shape: (N, 2) if any detections

        # Get predictions for all existing larvae.
        track_ids = list(self.larvae.keys())
        predictions = []
        for tid in track_ids:
            larva = self.larvae[tid]
            pred_pos = larva.predict()  # 2D point
            predictions.append(pred_pos)
        if predictions:
            predictions = np.array(predictions)  # Shape: (M, 2)
        else:
            predictions = np.empty((0, 2), np.float32)

        # If no tracks exist, initialize one for each detection.
        if predictions.shape[0] == 0:
            for i, measurement in enumerate(detection_measurements):
                new_larva = RobustLarva(self.next_id, tuple(measurement))
                new_larva.update_state(measurement, detection_endpoints[i])
                self.larvae[self.next_id] = new_larva
                self.next_id += 1
            return

        # If no detections exist, increment missed_frames for all tracks and exit.
        if detection_measurements.size == 0:
            for tid in track_ids:
                self.larvae[tid].missed_frames += 1
            return

        # Compute the cost matrix between predicted positions and detection measurements.
        cost_matrix = np.linalg.norm(
            predictions[:, np.newaxis, :] - detection_measurements[np.newaxis, :, :],
            axis=2,
        )

        # Solve the assignment problem using the Hungarian algorithm.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_threshold = self.distance_threshold
        assigned_tracks = set()
        assigned_detections = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < assignment_threshold:
                track_id = track_ids[i]
                larva = self.larvae[track_id]
                measurement = detection_measurements[j]
                endpoints = detection_endpoints[j]
                larva.update_state(measurement, endpoints)
                assigned_tracks.add(track_id)
                assigned_detections.add(j)

        # For tracks that were not assigned any detection, increment their missed frame counter.
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.larvae[tid].missed_frames += 1

        # Create new tracks for detections that were not assigned.
        for j in range(len(detection_measurements)):
            if j not in assigned_detections:
                measurement = detection_measurements[j]
                new_larva = RobustLarva(self.next_id, tuple(measurement))
                new_larva.update_state(measurement, detection_endpoints[j])
                self.larvae[self.next_id] = new_larva
                self.next_id += 1

        # Optionally remove tracks that have been lost for too long.
        remove_ids = [
            tid for tid, larva in self.larvae.items() if larva.missed_frames > 50
        ]
        for tid in remove_ids:
            del self.larvae[tid]

    def get_trajectories(self):
        """
        Return a list of dictionaries for each larva containing its id, track (list of positions),
        and head/tail information.
        """
        trajectories = []
        for larva in self.larvae.values():
            traj = {
                "id": larva.id,
                "track": [tuple(pos) for pos, conf in larva.track_buffer],
                "head": larva.head,
                "tail": larva.tail,
            }
            trajectories.append(traj)
        return trajectories


# ----------------------------
# Batch processing
# ----------------------------


class BatchProcessor:
    """
    Processes a batch of frames with bidirectional (forward and reverse) tracking.
    Forward and backward trajectories are merged using a simple Hungarian algorithm.
    """

    def __init__(self, batch_size: int, background):
        self.batch_size = batch_size
        self.background = background

    def process_batch(self, frames: List[np.ndarray]) -> List[dict]:
        # Forward pass tracking.
        tracker_fwd = RobustLarvaTracker(distance_threshold=50)
        for frame in frames:
            mask = process_frame(frame, self.background)
            detections = extract_objects(mask)
            tracker_fwd.update(detections)
        forward_tracks = tracker_fwd.get_trajectories()

        # Backward pass tracking (process frames in reverse order).
        tracker_bwd = RobustLarvaTracker(distance_threshold=50)
        for frame in frames[::-1]:
            mask = process_frame(frame, self.background)
            detections = extract_objects(mask)
            tracker_bwd.update(detections)
        backward_tracks = tracker_bwd.get_trajectories()
        # Reverse the backward trajectories so that time order matches.
        for traj in backward_tracks:
            traj["track"] = traj["track"][::-1]
        # Merge the two sets of trajectories.
        resolved_tracks = self._resolve_trajectories(forward_tracks, backward_tracks)
        return resolved_tracks

    def _resolve_trajectories(
        self, forward: List[dict], backward: List[dict]
    ) -> List[dict]:
        """
        For each forward trajectory, use its final position and the corresponding
        backward trajectory’s first position to compute a cost matrix. Merge those
        whose distance is below a threshold.
        """
        f_coords = [traj["track"][-1] for traj in forward if traj["track"]]
        b_coords = [traj["track"][0] for traj in backward if traj["track"]]
        if not f_coords or not b_coords:
            return forward
        cost_matrix = np.zeros((len(f_coords), len(b_coords)))
        for i, f in enumerate(f_coords):
            for j, b in enumerate(b_coords):
                cost_matrix[i, j] = np.linalg.norm(np.array(f) - np.array(b))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        merged = []
        used_backward = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 100:  # distance threshold
                merged.append(self._merge_trajectories(forward[i], backward[j]))
                used_backward.add(j)
            else:
                merged.append(forward[i])
        # Add any unmatched backward trajectories.
        for idx, traj in enumerate(backward):
            if idx not in used_backward:
                merged.append(traj)
        return merged

    def _merge_trajectories(self, f_traj: dict, b_traj: dict) -> dict:
        """
        Merge two trajectories (assumed to span the same batch) by averaging each
        corresponding point. Head and tail assignments are taken from whichever is available.
        """
        merged_track = []
        length = min(len(f_traj["track"]), len(b_traj["track"]))
        for k in range(length):
            pos = (np.array(f_traj["track"][k]) + np.array(b_traj["track"][k])) / 2
            merged_track.append(tuple(pos))
        merged = {
            "id": f_traj["id"],
            "track": merged_track,
            "head": f_traj["head"] if f_traj["head"] is not None else b_traj["head"],
            "tail": f_traj["tail"] if f_traj["tail"] is not None else b_traj["tail"],
        }
        return merged


# ----------------------------
# Drawing helper
# ----------------------------


def draw_tracking_on_frame(frame, trajectories: List[dict]):
    """
    Draws each trajectory (its path, head and tail markers, and ID) on the frame.
    Coordinates are converted to integers before drawing.
    """
    for traj in trajectories:
        track = traj["track"]
        if len(track) > 1:
            pts = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
        centroid = track[-1] if track else (0, 0)
        # Convert centroid to integer tuple
        cv2.putText(
            frame,
            f"ID:{traj['id']}",
            tuple(map(int, centroid)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        if traj.get("head"):
            cv2.circle(frame, tuple(map(int, traj["head"])), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "Head",
                tuple(map(int, traj["head"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        if traj.get("tail"):
            cv2.circle(frame, tuple(map(int, traj["tail"])), 5, (255, 255, 0), -1)
            cv2.putText(
                frame,
                "Tail",
                tuple(map(int, traj["tail"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
    return frame


# ----------------------------
# Asynchronous video processing
# ----------------------------


class AsyncVideoProcessor(threading.Thread):
    """
    Reads frames from the video asynchronously in batches (buffered),
    performs bidirectional tracking on each batch and sends the result
    (and a preview frame) to a callback function.
    """

    def __init__(
        self, video_path: str, callback, status_callback, buffer_size: int = 90
    ):
        super().__init__()
        self.video_path = video_path
        self.callback = callback  # Function to call with (results, preview_frame)
        self.status_callback = status_callback  # For updating GUI status
        self.buffer_size = buffer_size
        self.running = threading.Event()

    def run(self):
        self.running.set()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            self.status_callback("Error: Could not open video file.")
            return
        background = compute_background(self.video_path, num_frames=30)
        if background is None:
            print("Error: Could not compute background model.")
            self.status_callback("Error: Could not compute background model.")
            return
        batch_processor = BatchProcessor(
            batch_size=self.buffer_size, background=background
        )
        frame_count = 0
        while self.running.is_set():
            frames = []
            for _ in range(self.buffer_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            if frames:
                results = batch_processor.process_batch(frames)
                # Use the last frame of the batch for preview overlay.
                preview_frame = frames[-1].copy()
                preview_frame = draw_tracking_on_frame(preview_frame, results)
                frame_count += len(frames)
                self.status_callback(f"Processed {frame_count} frames.")
                self.callback(results, preview_frame)
                # Sleep a little to yield time to the GUI
                time.sleep(0.01)
            else:
                print("No more frames to process.")
                self.status_callback("Finished processing video.")
                break
        cap.release()
        print("Video processing thread ending.")

    def stop(self):
        self.running.clear()


# ----------------------------
# Tkinter GUI
# ----------------------------


class TrackerGUI:
    """
    A simple GUI using Tkinter that allows you to select a video,
    start tracking (which runs asynchronously), and stop tracking.
    """

    def __init__(self, master):
        self.master = master
        master.title("Robust Larva Tracker")
        master.geometry("400x250")
        self.video_path = None
        self.processor_thread = None

        self.select_button = tk.Button(
            master, text="Select Video", command=self.select_video
        )
        self.select_button.pack(pady=10)

        self.start_button = tk.Button(
            master,
            text="Start Tracking",
            command=self.start_tracking,
            state=tk.DISABLED,
        )
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(
            master, text="Stop Tracking", command=self.stop_tracking, state=tk.DISABLED
        )
        self.stop_button.pack(pady=10)

        self.status_label = tk.Label(master, text="Status: Waiting for video selection")
        self.status_label.pack(pady=10)

        self.preview_label = tk.Label(
            master, text="Preview will appear in a separate window."
        )
        self.preview_label.pack(pady=10)

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if path:
            self.video_path = path
            self.status_label.config(text=f"Selected: {path}")
            self.start_button.config(state=tk.NORMAL)
            print(f"Video selected: {path}")

    def start_tracking(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected.")
            return
        self.status_label.config(text="Tracking started...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.processor_thread = AsyncVideoProcessor(
            self.video_path, self.update_preview, self.update_status, buffer_size=90
        )
        self.processor_thread.start()
        print("Tracking thread started.")

    def stop_tracking(self):
        if self.processor_thread:
            self.processor_thread.stop()
            self.processor_thread.join(timeout=2)
            self.processor_thread = None
            self.status_label.config(text="Tracking stopped.")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            print("Tracking stopped.")

    def update_status(self, message):
        """
        Callback to update the status label from the processing thread.
        Since this may be called from another thread, use after() to schedule on main thread.
        """
        self.master.after(0, lambda: self.status_label.config(text=message))
        print(message)

    def update_preview(self, results, frame):
        """
        Callback function that receives the tracking results and preview frame
        from the AsyncVideoProcessor.
        We use after() to schedule the update on the main thread.
        """

        def show_frame():
            cv2.imshow("Larva Tracking Preview", frame)
            cv2.waitKey(1)

        self.master.after(0, show_frame)


def main():
    root = tk.Tk()
    gui = TrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (gui.stop_tracking(), root.destroy()))
    root.mainloop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
