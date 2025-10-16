import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
from pathlib import Path

class TrajectoryExtractor:
    def __init__(self):
        self.image = None
        self.trajectories = []
        self.predicted_tracks = 5
        self.clustering_eps = 10
        self.min_samples = 10  # Increased to reduce over-clustering
        self.min_trajectory_length = 15  # Minimum points for a valid trajectory
    
        
    def load_image(self, filepath):
        """Load and preprocess the PNG image"""
        try:
            self.image = cv2.imread(filepath)
            if self.image is None:
                raise ValueError("Could not load image")
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def extract_color_gradients(self):
        """Extract color gradients representing temporal progression"""
        if self.image is None:
            return []
        
        # Convert to HSV for better color analysis
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        
        # Create a mask to ignore very dark pixels (background)
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        mask = gray > 20  # Threshold to ignore dark background
        
        # Extract pixels with color information
        colored_pixels = []
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                if mask[y, x]:
                    # Get RGB and HSV values
                    rgb = self.image[y, x]
                    hsv = hsv_image[y, x]
                    
                    # Calculate color intensity as proxy for temporal position
                    # Assuming brighter/more saturated = later in time
                    temporal_weight = (rgb.sum() + hsv[1] + hsv[2]) / 5.0
                    
                    colored_pixels.append({
                        'x': x,
                        'y': y,
                        'rgb': rgb,
                        'hsv': hsv,
                        'temporal_weight': temporal_weight,
                        'hue': hsv[0]
                    })
        
        return colored_pixels
    
    def cluster_trajectory_points(self, colored_pixels):
        """Cluster pixels into trajectory segments with improved algorithm"""
        if not colored_pixels:
            return []
        
        # Prepare data for clustering - focus more on spatial proximity
        positions = np.array([[p['x'], p['y']] for p in colored_pixels])
        temporal_weights = np.array([p['temporal_weight'] for p in colored_pixels])
        hues = np.array([p['hue'] for p in colored_pixels])
        
        # Weight spatial features more heavily than color features
        # This prevents over-segmentation based on minor color variations
        spatial_weight = 3.0
        temporal_weight = 0.5
        color_weight = 0.3
        
        # Normalize hues (handle circular nature)
        hues_norm = np.column_stack([np.cos(hues * np.pi / 90), np.sin(hues * np.pi / 90)])
        
        # Combine features with different weights
        features = np.column_stack([
            positions * spatial_weight,
            temporal_weights.reshape(-1, 1) * temporal_weight,
            hues_norm * color_weight
        ])
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply DBSCAN clustering with adjusted parameters
        eps_adjusted = self.clustering_eps / 50  # More conservative clustering
        clustering = DBSCAN(eps=eps_adjusted, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(features_scaled)
        
        # Group pixels by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(colored_pixels[i])
        
        # Filter out very small clusters
        filtered_clusters = {k: v for k, v in clusters.items() if len(v) >= self.min_trajectory_length}
        
        return filtered_clusters
    
    def extract_trajectories_from_clusters(self, clusters):
        """Extract ordered trajectories from clustered points with consolidation"""
        trajectories = []
        
        print(f"Processing {len(clusters)} clusters...")
        
        for cluster_id, points in clusters.items():
            if len(points) < self.min_trajectory_length:  # Skip small clusters
                continue
            
            try:
                # Sort points by temporal weight (time progression)
                points_sorted = sorted(points, key=lambda p: p['temporal_weight'])
                
                # Extract trajectory as sequence of (x, y) coordinates
                trajectory = [(p['x'], p['y']) for p in points_sorted]
                
                if not trajectory:  # Safety check
                    continue
                
                # Smooth trajectory to reduce noise
                trajectory = self.smooth_trajectory(trajectory)
                
                if not trajectory:  # Safety check after smoothing
                    continue
                
                # Check for interruptions but be more lenient
                trajectory_segments = self.handle_interruptions(trajectory, max_gap=80)
                
                # Only keep segments that are long enough
                valid_segments = [seg for seg in trajectory_segments if seg and len(seg) >= self.min_trajectory_length]
                
                trajectories.extend(valid_segments)
                
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
                continue
        
        print(f"Found {len(trajectories)} initial trajectory segments")
        
        # Post-process: merge nearby trajectories that might belong together
        try:
            trajectories = self.merge_nearby_trajectories(trajectories)
            print(f"After merging: {len(trajectories)} trajectory segments")
        except Exception as e:
            print(f"Error during trajectory merging: {e}")
        
        # Sort by length (longest first) and take top N if we have too many
        trajectories.sort(key=len, reverse=True)
        
        if len(trajectories) > self.predicted_tracks * 3:  # If we have way too many
            trajectories = trajectories[:self.predicted_tracks * 3]
            print(f"Limiting to top {len(trajectories)} longest trajectories")
        
        return trajectories
    
    def smooth_trajectory(self, trajectory, window_size=3):
        """Apply smoothing to reduce noise in trajectory"""
        if len(trajectory) < window_size:
            return trajectory
        
        trajectory_array = np.array(trajectory)
        smoothed = []
        
        for i in range(len(trajectory)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(trajectory), i + window_size // 2 + 1)
            
            # Average coordinates in the window
            window_points = trajectory_array[start_idx:end_idx]
            smoothed_point = np.mean(window_points, axis=0)
            smoothed.append(tuple(smoothed_point))
        return smoothed
        
    def merge_nearby_trajectories(self, trajectories):
        """Merge trajectories that might be parts of the same track"""
        if len(trajectories) <= 1:
            return trajectories
        
        merged = []
        used = set()
        
        try:
            for i, traj1 in enumerate(trajectories):
                if i in used or not traj1:
                    continue
                    
                current_traj = list(traj1)
                used.add(i)
                
                # Look for trajectories to merge with this one
                for j, traj2 in enumerate(trajectories):
                    if j <= i or j in used or not traj2:
                        continue
                    
                    # Check if trajectories can be connected
                    if self.can_merge_trajectories(current_traj, traj2):
                        # Merge them
                        current_traj = self.connect_trajectories(current_traj, traj2)
                        used.add(j)
                
                if current_traj:  # Only add non-empty trajectories
                    merged.append(current_traj)
        
        except Exception as e:
            print(f"Error in merge_nearby_trajectories: {e}")
            return [traj for traj in trajectories if traj]  # Return original, filtered for empty ones
        
        return merged
    
    def can_merge_trajectories(self, traj1, traj2, max_distance=100):
        """Check if trajectories can merge based on spatial+temporal constraints"""
        if not traj1 or not traj2 or len(traj1) < 2 or len(traj2) < 2:
            return False
            
        try:
            # Get endpoints and their colors
            endpoints = [
                (traj1[0], traj2[0], 'start-start'),   
                (traj1[0], traj2[-1], 'start-end'),  
                (traj1[-1], traj2[0], 'end-start'),  
                (traj1[-1], traj2[-1], 'end-end')
            ]
            
            for p1, p2, connection_type in endpoints:
                # Check spatial distance
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if dist > max_distance:
                    continue
                    
                # Get colors at endpoints
                color1 = self.image[int(p1[1]), int(p1[0])]
                color2 = self.image[int(p2[1]), int(p2[0])]
                
                # Convert to HSV for better color comparison
                hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0]
                hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0]
                
                # Check temporal progression (brighter/more saturated = later)
                temporal1 = (np.sum(color1) + hsv1[1] + hsv1[2]) / 5.0
                temporal2 = (np.sum(color2) + hsv2[1] + hsv2[2]) / 5.0
                
                # Only allow connections that follow temporal progression
                valid_progression = False
                if connection_type in ['start-start', 'start-end']:
                    valid_progression = temporal1 < temporal2
                else:
                    valid_progression = temporal2 < temporal1
                    
                if not valid_progression:
                    continue
                    
                # Check path for black pixels and color continuity
                num_samples = int(dist * 2)
                x_samples = np.linspace(p1[0], p2[0], num_samples, dtype=int)
                y_samples = np.linspace(p1[1], p2[1], num_samples, dtype=int)
                
                valid_path = True
                prev_temporal = temporal1
                
                for x, y in zip(x_samples, y_samples):
                    if (y < 0 or y >= self.image.shape[0] or 
                        x < 0 or x >= self.image.shape[1]):
                        valid_path = False
                        break
                        
                    # Check for black pixels and color progression
                    current_color = self.image[y, x]
                    if np.sum(current_color) < 20:  # Too dark
                        valid_path = False
                        break
                        
                    current_hsv = cv2.cvtColor(np.uint8([[current_color]]), cv2.COLOR_RGB2HSV)[0][0]
                    current_temporal = (np.sum(current_color) + current_hsv[1] + current_hsv[2]) / 5.0
                    
                    # Ensure temporal progression along path
                    if abs(current_temporal - prev_temporal) > 50:  # Allow small variations
                        valid_path = False
                        break
                        
                    prev_temporal = current_temporal
                
                if valid_path:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error in can_merge_trajectories: {e}")
            return False
    def connect_trajectories(self, traj1, traj2):
        """Connect two trajectories in the best order"""
        if not traj1 or not traj2:
            return traj1 if traj1 else traj2
            
        try:
            distances = [
                (np.sqrt((traj1[0][0] - traj2[0][0])**2 + (traj1[0][1] - traj2[0][1])**2), 'start-start'),
                (np.sqrt((traj1[0][0] - traj2[-1][0])**2 + (traj1[0][1] - traj2[-1][1])**2), 'start-end'),
                (np.sqrt((traj1[-1][0] - traj2[0][0])**2 + (traj1[-1][1] - traj2[0][1])**2), 'end-start'),
                (np.sqrt((traj1[-1][0] - traj2[-1][0])**2 + (traj1[-1][1] - traj2[-1][1])**2), 'end-end')
            ]
            
            min_dist, connection_type = min(distances)
            
            if connection_type == 'start-start':
                return list(reversed(traj2)) + traj1
            elif connection_type == 'start-end':
                return traj2 + traj1
            elif connection_type == 'end-start':
                return traj1 + traj2
            else:  # end-end
                return traj1 + list(reversed(traj2))
        except Exception as e:
            print(f"Error in connect_trajectories: {e}")
            return traj1  # Return first trajectory if connection fails
    
    def handle_interruptions(self, trajectory, max_gap=80):
        """Detect and handle interruptions in trajectories with more lenient gap detection"""
        if not trajectory or len(trajectory) < 2:
            return [trajectory] if trajectory else []
        
        segments = []
        current_segment = [trajectory[0]]
        
        try:
            for i in range(1, len(trajectory)):
                prev_point = trajectory[i-1]
                curr_point = trajectory[i]
                
                # Calculate distance between consecutive points
                distance = np.sqrt((curr_point[0] - prev_point[0])**2 + 
                                 (curr_point[1] - prev_point[1])**2)
                
                if distance > max_gap:
                    # Interruption detected, start new segment
                    if len(current_segment) >= self.min_trajectory_length:
                        segments.append(current_segment)
                    current_segment = [curr_point]
                else:
                    current_segment.append(curr_point)
            
            # Add the last segment
            if len(current_segment) >= self.min_trajectory_length:
                segments.append(current_segment)
        
        except Exception as e:
            print(f"Error in handle_interruptions: {e}")
            return [trajectory] if len(trajectory) >= self.min_trajectory_length else []
        
        return segments
    
    def process_image(self):
        """Main processing pipeline"""
        if self.image is None:
            return False
        
        print("Extracting color gradients...")
        colored_pixels = self.extract_color_gradients()
        
        if not colored_pixels:
            print("No colored pixels found!")
            return False
        
        print(f"Found {len(colored_pixels)} colored pixels")
        
        print("Clustering trajectory points...")
        clusters = self.cluster_trajectory_points(colored_pixels)
        
        print(f"Found {len(clusters)} clusters")
        
        print("Extracting trajectories...")
        self.trajectories = self.extract_trajectories_from_clusters(clusters)
        
        print(f"Extracted {len(self.trajectories)} trajectory segments")
        
        return len(self.trajectories) > 0
    
    def convert_to_trackpy_format(self):
        """Convert trajectories to trackpy-compatible numpy array"""
        if not self.trajectories:
            return None
        
        all_points = []
        particle_id = 0
        
        for trajectory in self.trajectories:
            for frame, (x, y) in enumerate(trajectory):
                all_points.append([x, y, frame, particle_id])
            particle_id += 1
        
        # Convert to numpy array with column names compatible with trackpy
        trajectory_array = np.array(all_points)
        
        return trajectory_array
    
    def plot_trajectories(self):
        """Plot detected trajectories as scatter points without connections"""
        if not self.trajectories:
            print("No trajectories to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Show original image as background
        if self.image is not None:
            plt.imshow(self.image, alpha=0.3)
        
        # Plot each trajectory with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.trajectories)))
        
        for i, trajectory in enumerate(self.trajectories):
            if len(trajectory) > 1:
                x_coords = [point[0] for point in trajectory]
                y_coords = [point[1] for point in trajectory]
                
                # Plot only points, no lines ('o' instead of 'o-')
                plt.scatter(x_coords, y_coords, 
                        color=colors[i], s=25,  # s=marker size
                        label=f'Track {i+1} ({len(trajectory)} points)')
                
                # Mark start and end points
                plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, alpha=0.7)
                plt.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, alpha=0.7)
        
        plt.title(f'Detected Trajectories ({len(self.trajectories)} tracks)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.tight_layout()
        plt.show()

class TrajectoryGUI:
    def __init__(self):
        self.extractor = TrajectoryExtractor()
        self.root = tk.Tk()
        self.root.title("Larval Track Trajectory Extractor")
        self.root.geometry("400x300")
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # File selection
        tk.Label(self.root, text="Larval Track Trajectory Extractor", 
                font=("Arial", 14, "bold")).pack(pady=10)
        
        tk.Button(self.root, text="Select PNG Image", 
                 command=self.select_image, width=20, height=2).pack(pady=5)
        
        # Parameters
        params_frame = tk.Frame(self.root)
        params_frame.pack(pady=10)
        
        tk.Label(params_frame, text="Predicted number of tracks:").pack()
        self.tracks_var = tk.StringVar(value="5")
        tk.Entry(params_frame, textvariable=self.tracks_var, width=10).pack(pady=2)
        
        tk.Label(params_frame, text="Min trajectory length:").pack()
        self.min_length_var = tk.StringVar(value="15")
        tk.Entry(params_frame, textvariable=self.min_length_var, width=10).pack(pady=2)
        
        tk.Label(params_frame, text="Min samples per cluster:").pack()
        self.min_samples_var = tk.StringVar(value="10")
        tk.Entry(params_frame, textvariable=self.min_samples_var, width=10).pack(pady=2)
        
        # Processing buttons
        tk.Button(self.root, text="Process Image", 
                 command=self.process_image, width=20, height=2).pack(pady=5)
        
        tk.Button(self.root, text="Show Results", 
                 command=self.show_results, width=20, height=2).pack(pady=5)
        
        tk.Button(self.root, text="Save Trajectories", 
                 command=self.save_trajectories, width=20, height=2).pack(pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready", 
                                   relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        """File dialog for image selection"""
        filepath = filedialog.askopenfilename(
            title="Select PNG Image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.extractor.load_image(filepath):
                self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
                self.image_path = filepath
            else:
                messagebox.showerror("Error", "Failed to load image")
                self.status_label.config(text="Error loading image")
    
    def process_image(self):
        """Process the loaded image"""
        if self.extractor.image is None:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            # Update parameters
            self.extractor.predicted_tracks = int(self.tracks_var.get())
            self.extractor.min_trajectory_length = int(self.min_length_var.get())
            self.extractor.min_samples = int(self.min_samples_var.get())
            
            self.status_label.config(text="Processing...")
            self.root.update()
            
            success = self.extractor.process_image()
            
            if success:
                num_tracks = len(self.extractor.trajectories)
                self.status_label.config(text=f"Found {num_tracks} trajectory segments")
                
                if abs(num_tracks - self.extractor.predicted_tracks) > 2:
                    messagebox.showinfo("Info", 
                        f"Found {num_tracks} tracks, expected ~{self.extractor.predicted_tracks}. "
                        "Consider adjusting clustering sensitivity.")
            else:
                self.status_label.config(text="Processing failed")
                messagebox.showerror("Error", "Failed to extract trajectories")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter values: {e}")
    
    def show_results(self):
        """Display the trajectory plot"""
        if not self.extractor.trajectories:
            messagebox.showwarning("Warning", "No trajectories to display. Process image first.")
            return
        
        self.extractor.plot_trajectories()
    
    def save_trajectories(self):
        """Save trajectories to file"""
        if not self.extractor.trajectories:
            messagebox.showwarning("Warning", "No trajectories to save. Process image first.")
            return
        
        # Get save location
        filepath = filedialog.asksaveasfilename(
            title="Save Trajectories",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            trajectory_array = self.extractor.convert_to_trackpy_format()
            
            if filepath.endswith('.csv'):
                # Save as CSV with headers
                header = "x,y,frame,particle"
                np.savetxt(filepath, trajectory_array, delimiter=',', header=header, comments='')
            else:
                # Save as NumPy array
                np.save(filepath, trajectory_array)
            
            self.status_label.config(text=f"Saved: {os.path.basename(filepath)}")
            messagebox.showinfo("Success", f"Trajectories saved to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save trajectories: {e}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function to run the application"""
    print("Starting Larval Track Trajectory Extractor...")
    app = TrajectoryGUI()
    app.run()

if __name__ == "__main__":
    main()