'''This is based on the idea that a video of white larvae on a black background can be binarized, projected, and viewed as a 'map' like a video game.
It's my belief that a maze solving algorithm, like what's used for NPCs in video games to solve paths, will provide a low CPU/GPU method for solving the 'tracks' of the larvae'''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from tkinter import filedialog, messagebox, simpledialog
import tkinter as tk
from collections import deque
import colorsys
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import itertools

class TemporalPathTracer:
    def __init__(self):
        self.temporal_image = None
        self.scale_bar_image = None
        self.paths = []
        self.animation_active = False
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.click_mode = None
        self.cid = None
        self.scale_line_points = []
        self.temporal_colors = {'start': [], 'middle': [], 'end': []}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Add buttons
        ax_load_temporal = plt.axes([0.02, 0.95, 0.10, 0.03])
        ax_load_scale = plt.axes([0.13, 0.95, 0.10, 0.03])
        ax_draw_line = plt.axes([0.24, 0.95, 0.10, 0.03])
        ax_find_paths = plt.axes([0.35, 0.95, 0.10, 0.03])
        ax_animate = plt.axes([0.46, 0.95, 0.08, 0.03])
        ax_clear = plt.axes([0.55, 0.95, 0.08, 0.03])
        
        self.btn_load_temporal = Button(ax_load_temporal, 'Load Temporal')
        self.btn_load_scale = Button(ax_load_scale, 'Load Scale')
        self.btn_draw_line = Button(ax_draw_line, 'Draw Line')
        self.btn_find_paths = Button(ax_find_paths, 'Find Paths')
        self.btn_animate = Button(ax_animate, 'Animate')
        self.btn_clear = Button(ax_clear, 'Clear')
        
        # Initially disable some buttons
        self.btn_draw_line.ax.set_facecolor('lightgray')
        self.btn_find_paths.ax.set_facecolor('lightgray')
        
        # Connect button events
        self.btn_load_temporal.on_clicked(self.load_temporal_image)
        self.btn_load_scale.on_clicked(self.load_scale_bar)
        self.btn_draw_line.on_clicked(self.start_line_drawing)
        self.btn_find_paths.on_clicked(self.find_all_paths)
        self.btn_animate.on_clicked(self.start_animation)
        self.btn_clear.on_clicked(self.clear_all)
        
        # Set up subplots
        self.axes[0,0].set_title('Temporal Image')
        self.axes[0,1].set_title('Color Intensity Map')
        self.axes[1,0].set_title('Scale Bar - Draw line from start to end')
        self.axes[1,1].set_title('Path Results')
        
        for ax in self.axes.flat:
            ax.axis('off')
        
        # Add status text
        self.status_text = self.fig.text(0.5, 0.02, 'Load temporal image to begin', 
                                        ha='center', fontsize=12, 
                                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
    def update_status(self, message):
        """Update status message"""
        self.status_text.set_text(message)
        plt.draw()
        
    def disconnect_click_events(self):
        """Safely disconnect any existing click events"""
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
            
    def load_temporal_image(self, event=None):
        """Load the temporal color-coded image"""
        root = tk.Tk()
        root.withdraw()
        
        try:
            file_path = filedialog.askopenfilename(
                title="Select Temporal Color Image",
                filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            self.temporal_image = cv2.imread(file_path)
            if self.temporal_image is None:
                messagebox.showerror("Error", "Could not load temporal image")
                return
                
            self.temporal_image = cv2.cvtColor(self.temporal_image, cv2.COLOR_BGR2RGB)
            
            # Display temporal image
            self.axes[0,0].clear()
            self.axes[0,0].imshow(self.temporal_image)
            self.axes[0,0].set_title('Temporal Image')
            self.axes[0,0].axis('off')
            
            # Create and display intensity map
            self.create_intensity_map()
            
            plt.draw()
            self.update_status("Temporal image loaded. Now load the scale bar image.")
            
        except Exception as e:
            print(f"Error loading temporal image: {e}")
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
        finally:
            root.destroy()
    
    def create_intensity_map(self):
        """Create an intensity map based on color brightness/saturation"""
        # Convert to HSV to better analyze color intensity
        hsv_image = cv2.cvtColor(self.temporal_image, cv2.COLOR_RGB2HSV)
        
        # Use saturation and value to create intensity map
        saturation = hsv_image[:,:,1].astype(float) / 255.0
        value = hsv_image[:,:,2].astype(float) / 255.0
        
        # Combine saturation and value for intensity
        intensity_map = (saturation * value)
        
        self.axes[0,1].clear()
        self.axes[0,1].imshow(intensity_map, cmap='hot')
        self.axes[0,1].set_title('Color Intensity Map')
        self.axes[0,1].axis('off')
    
    def load_scale_bar(self, event=None):
        """Load the scale bar image"""
        root = tk.Tk()
        root.withdraw()
        
        try:
            file_path = filedialog.askopenfilename(
                title="Select Scale Bar Image",
                filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            self.scale_bar_image = cv2.imread(file_path)
            if self.scale_bar_image is None:
                messagebox.showwarning("Warning", "Could not load scale bar image")
                return
                
            self.scale_bar_image = cv2.cvtColor(self.scale_bar_image, cv2.COLOR_BGR2RGB)
            
            self.axes[1,0].clear()
            self.axes[1,0].imshow(self.scale_bar_image)
            self.axes[1,0].set_title('Scale Bar - Draw line from start to end')
            self.axes[1,0].axis('off')
            
            # Enable line drawing
            self.btn_draw_line.ax.set_facecolor('lightgreen')
            
            plt.draw()
            self.update_status("Scale bar loaded. Click 'Draw Line' then draw a line across the scale bar.")
            
        except Exception as e:
            print(f"Error loading scale bar: {e}")
            messagebox.showerror("Error", f"Error loading image: {str(e)}")
        finally:
            root.destroy()
    
    def start_line_drawing(self, event=None):
        """Start line drawing mode on scale bar"""
        if self.scale_bar_image is None:
            messagebox.showwarning("Warning", "Please load scale bar image first.")
            return
            
        self.scale_line_points = []
        self.click_mode = 'draw_line'
        self.disconnect_click_events()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update_status("Click two points on the scale bar to draw a line from temporal start to end.")
    
    def on_click(self, event):
        """Handle click events based on current mode"""
        try:
            if event is None or event.xdata is None or event.ydata is None:
                return
                
            if self.click_mode == 'draw_line' and event.inaxes == self.axes[1,0]:
                self.handle_line_drawing(event)
                
        except Exception as e:
            print(f"Error in click handler: {e}")
    
    def handle_line_drawing(self, event):
        """Handle line drawing on scale bar"""
        try:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            
            if (x < 0 or x >= self.scale_bar_image.shape[1] or 
                y < 0 or y >= self.scale_bar_image.shape[0]):
                return
            
            # Add point to line
            self.scale_line_points.append((x, y))
            
            # Draw point
            self.axes[1,0].plot(x, y, 'ro', markersize=8)
            
            if len(self.scale_line_points) == 1:
                self.update_status("Click the end point of the temporal scale line.")
                
            elif len(self.scale_line_points) == 2:
                # Draw line
                x_coords = [p[0] for p in self.scale_line_points]
                y_coords = [p[1] for p in self.scale_line_points]
                self.axes[1,0].plot(x_coords, y_coords, 'r-', linewidth=3)
                
                # Process the line
                self.process_scale_line()
                
                # Disable line drawing, enable path finding
                self.disconnect_click_events()
                self.btn_draw_line.ax.set_facecolor('lightgray')
                self.btn_find_paths.ax.set_facecolor('lightgreen')
                
                plt.draw()
                self.update_status("Scale line drawn. Click 'Find Paths' to trace temporal gradients.")
            
            plt.draw()
            
        except Exception as e:
            print(f"Error handling line drawing: {e}")
    
    def process_scale_line(self):
        """Process the drawn line to extract temporal colors"""
        try:
            start_point, end_point = self.scale_line_points
            
            # Get more sample points for better color resolution
            num_samples = 50
            
            # Calculate line points
            line_points = []
            for i in range(num_samples):
                t = i / (num_samples - 1)
                x = int(start_point[0] + t * (end_point[0] - start_point[0]))
                y = int(start_point[1] + t * (end_point[1] - start_point[1]))
                
                # Get color at this point
                color = self.scale_bar_image[y, x]
                line_points.append((x, y, color))
            
            # Divide into more granular sections for better gradient following
            third = len(line_points) // 3
            
            self.temporal_colors = {
                'start': [point[2] for point in line_points[:third]],
                'middle': [point[2] for point in line_points[third:2*third]],
                'end': [point[2] for point in line_points[2*third:]]
            }
            
            print(f"Extracted temporal colors:")
            print(f"  Start: {len(self.temporal_colors['start'])} colors")
            print(f"  Middle: {len(self.temporal_colors['middle'])} colors")
            print(f"  End: {len(self.temporal_colors['end'])} colors")
            
            # Visualize the divisions on the scale bar
            for i, (x, y, color) in enumerate(line_points):
                if i < third:
                    marker_color = 'green'  # Start
                elif i < 2*third:
                    marker_color = 'yellow'  # Middle
                else:
                    marker_color = 'red'  # End
                    
                self.axes[1,0].plot(x, y, 'o', color=marker_color, markersize=3)
            
        except Exception as e:
            print(f"Error processing scale line: {e}")
    
    def find_all_paths(self, event=None):
        """Find paths following temporal color gradients"""
        if not self.temporal_colors['start'] or not self.temporal_colors['end']:
            messagebox.showwarning("Warning", "No temporal colors extracted. Please draw line on scale bar first.")
            return
        
        if self.temporal_image is None:
            messagebox.showwarning("Warning", "Please load temporal image first.")
            return
        
        try:
            self.paths = []
            self.update_status("Finding temporal gradient paths...")
            plt.pause(0.1)
            
            # Find all start points with more liberal tolerance
            start_points = self.find_points_for_colors(self.temporal_colors['start'], tolerance=15)
            middle_points = self.find_points_for_colors(self.temporal_colors['middle'], tolerance=15)
            end_points = self.find_points_for_colors(self.temporal_colors['end'], tolerance=15)
            
            print(f"Found temporal points:")
            print(f"  Start points: {len(start_points)}")
            print(f"  Middle points: {len(middle_points)}")
            print(f"  End points: {len(end_points)}")
            
            if not start_points:
                messagebox.showwarning("Warning", "No start points found. Try adjusting color tolerance.")
                return
            
            # Find gradient-following paths
            path_id = 0
            
            # Direct start to end paths following gradient
            for start in start_points[:10]:  # Limit to prevent too many combinations
                for end in end_points[:10]:
                    self.update_status(f"Finding gradient path {path_id + 1}...")
                    plt.pause(0.01)
                    
                    path = self.find_gradient_path(start, end)
                    if path and len(path) > 5:  # Only keep substantial paths
                        self.paths.append({
                            'path': path,
                            'color': plt.cm.viridis(path_id / 20)[:3],
                            'type': 'gradient',
                            'start': start,
                            'end': end,
                            'id': path_id
                        })
                        path_id += 1
            
            # Multi-stage paths through middle points
            for start in start_points[:5]:
                for middle in middle_points[:5]:
                    for end in end_points[:5]:
                        self.update_status(f"Finding multi-stage path {path_id + 1}...")
                        plt.pause(0.01)
                        
                        path1 = self.find_gradient_path(start, middle)
                        path2 = self.find_gradient_path(middle, end)
                        
                        if path1 and path2:
                            # Combine paths
                            combined_path = path1 + path2[1:]  # Avoid duplicate middle point
                            self.paths.append({
                                'path': combined_path,
                                'color': plt.cm.plasma(path_id / 20)[:3],
                                'type': 'multi_stage',
                                'start': start,
                                'middle': middle,
                                'end': end,
                                'id': path_id
                            })
                            path_id += 1
                        
                        if path_id >= 20:  # Reasonable limit
                            break
                    if path_id >= 20:
                        break
                if path_id >= 20:
                    break
            
            # Display results
            self.display_path_results()
            
            self.update_status(f"Found {len(self.paths)} gradient paths. Click 'Animate' to see them.")
            
        except Exception as e:
            print(f"Error finding paths: {e}")
            messagebox.showerror("Error", f"Error finding paths: {str(e)}")
    
    def find_points_for_colors(self, color_list, tolerance=15):
        """Find points in temporal image that match any of the given colors with liberal tolerance"""
        all_points = []
        
        for target_color in color_list:
            # Find similar colors with increased tolerance
            color_diff = np.abs(self.temporal_image.astype(float) - target_color.astype(float))
            color_mask = np.all(color_diff <= tolerance, axis=2)
            
            # Get coordinates of matching points
            point_coords = np.where(color_mask)
            
            if len(point_coords[0]) > 0:
                points = list(zip(point_coords[1], point_coords[0]))  # (x, y) format
                all_points.extend(points)
        
        if not all_points:
            return []
        
        # Cluster nearby points to avoid redundancy
        if len(all_points) > 1:
            coords_array = np.array(all_points)
            clustering = DBSCAN(eps=30, min_samples=1).fit(coords_array)
            
            clustered_points = []
            for cluster_id in np.unique(clustering.labels_):
                cluster_points = coords_array[clustering.labels_ == cluster_id]
                centroid = np.mean(cluster_points, axis=0).astype(int)
                clustered_points.append(tuple(centroid))
            
            return clustered_points
        
        return all_points
    
    def find_gradient_path(self, start, goal):
        """Find path following color gradients towards the goal"""
        try:
            current_pos = start
            path = [current_pos]
            visited = {current_pos}
            max_steps = 500
            
            for step in range(max_steps):
                # Check if we've reached the goal
                if abs(current_pos[0] - goal[0]) < 5 and abs(current_pos[1] - goal[1]) < 5:
                    path.append(goal)
                    return path
                
                # Get neighbors and evaluate them
                neighbors = self.get_gradient_neighbors(current_pos, goal, visited)
                
                if not neighbors:
                    # If stuck, try a broader search
                    neighbors = self.get_any_neighbors(current_pos, visited)
                
                if not neighbors:
                    break
                
                # Choose best neighbor based on gradient and distance
                best_neighbor = self.choose_best_neighbor(neighbors, current_pos, goal)
                
                if best_neighbor is None:
                    break
                
                path.append(best_neighbor)
                visited.add(best_neighbor)
                current_pos = best_neighbor
            
            # Return path if it made reasonable progress
            if len(path) > 10:
                return path
            
            return None
            
        except Exception as e:
            print(f"Error in gradient pathfinding: {e}")
            return None
    
    def get_gradient_neighbors(self, pos, goal, visited):
        """Get neighbors that follow a reasonable color gradient"""
        x, y = pos
        neighbors = []
        current_color = self.temporal_image[y, x]
        
        # Check 8-connected neighbors
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < self.temporal_image.shape[1] and 
                0 <= ny < self.temporal_image.shape[0] and
                (nx, ny) not in visited):
                
                neighbor_color = self.temporal_image[ny, nx]
                
                # Check if this neighbor represents a reasonable color transition
                color_distance = np.linalg.norm(neighbor_color.astype(float) - current_color.astype(float))
                
                # Allow reasonable color changes (not pure black/white transitions)
                if color_distance < 100 and not self.is_background_color(neighbor_color):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def get_any_neighbors(self, pos, visited):
        """Get any valid neighbors when gradient following fails"""
        x, y = pos
        neighbors = []
        
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < self.temporal_image.shape[1] and 
                0 <= ny < self.temporal_image.shape[0] and
                (nx, ny) not in visited):
                
                neighbor_color = self.temporal_image[ny, nx]
                if not self.is_background_color(neighbor_color):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def is_background_color(self, color):
        """Check if a color is likely background (very dark or very bright)"""
        brightness = np.mean(color)
        return brightness < 30 or brightness > 225
    
    def choose_best_neighbor(self, neighbors, current_pos, goal):
        """Choose the best neighbor based on gradient and distance to goal"""
        if not neighbors:
            return None
        
        best_neighbor = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            # Distance to goal (lower is better)
            distance_to_goal = np.sqrt((neighbor[0] - goal[0])**2 + (neighbor[1] - goal[1])**2)
            
            # Color similarity to goal colors (if we have them)
            color_score = self.get_color_progression_score(neighbor, goal)
            
            # Combined score
            score = distance_to_goal + color_score * 0.5
            
            if score < best_score:
                best_score = score
                best_neighbor = neighbor
        
        return best_neighbor
    
    def get_color_progression_score(self, pos, goal):
        """Score how well a position fits the temporal color progression"""
        try:
            pos_color = self.temporal_image[pos[1], pos[0]]
            
            # Compare with end colors to see if we're progressing in the right direction
            min_distance = float('inf')
            for end_color in self.temporal_colors['end']:
                distance = np.linalg.norm(pos_color.astype(float) - end_color.astype(float))
                min_distance = min(min_distance, distance)
            
            return min_distance
        except:
            return 100  # High penalty for invalid positions
    
    def display_path_results(self):
        """Display the found paths on the results subplot"""
        self.axes[1,1].clear()
        self.axes[1,1].imshow(self.temporal_image, alpha=0.7)
        self.axes[1,1].set_title(f'Gradient Paths Found ({len(self.paths)} paths)')
        self.axes[1,1].axis('off')
        
        # Draw all paths
        for i, path_data in enumerate(self.paths):
            path = path_data['path']
            color = path_data['color']
            
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            linewidth = 3 if path_data['type'] == 'gradient' else 2
            alpha = 0.8
            
            self.axes[1,1].plot(x_coords, y_coords, color=color, 
                               linewidth=linewidth, alpha=alpha, label=f"Path {i+1}")
            
            # Mark start and end points
            self.axes[1,1].plot(path[0][0], path[0][1], 'go', markersize=8)  # Green start
            self.axes[1,1].plot(path[-1][0], path[-1][1], 'ro', markersize=8)  # Red end
        
        plt.draw()
    
    def start_animation(self, event=None):
        """Animate all found paths"""
        if not self.paths:
            messagebox.showwarning("Warning", "No paths found. Please find paths first.")
            return
            
        self.animation_active = True
        self.animate_all_paths()
    
    def animate_all_paths(self):
        """Animate all paths simultaneously"""
        if not self.paths:
            return
        
        max_steps = max(len(p['path']) for p in self.paths) if self.paths else 0
        
        for step in range(max_steps):
            if not self.animation_active:
                break
                
            # Clear and redraw
            self.axes[1,1].clear()
            self.axes[1,1].imshow(self.temporal_image, alpha=0.7)
            self.axes[1,1].set_title(f'Animating Gradient Paths (Step {step+1}/{max_steps})')
            self.axes[1,1].axis('off')
            
            # Draw paths up to current step
            for path_data in self.paths:
                path = path_data['path']
                color = path_data['color']
                
                if step < len(path):
                    # Draw path up to current step
                    current_path = path[:step+1]
                    x_coords = [p[0] for p in current_path]
                    y_coords = [p[1] for p in current_path]
                    
                    self.axes[1,1].plot(x_coords, y_coords, color=color, 
                                       linewidth=3, alpha=0.8)
                    
                    # Draw current position
                    self.axes[1,1].plot(path[step][0], path[step][1], 'o', 
                                       color=color, markersize=8)
                else:
                    # Draw completed path
                    x_coords = [p[0] for p in path]
                    y_coords = [p[1] for p in path]
                    self.axes[1,1].plot(x_coords, y_coords, color=color, 
                                       linewidth=2, alpha=0.4)
            
            plt.draw()
            plt.pause(0.1)
        
        self.update_status(f"Animation complete. Traced {len(self.paths)} gradient paths.")
    
    def clear_all(self, event=None):
        """Clear all data and reset"""
        self.paths = []
        self.scale_line_points = []
        self.temporal_colors = {'start': [], 'middle': [], 'end': []}
        
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
        
        # Reset titles
        self.axes[0,0].set_title('Temporal Image')
        self.axes[0,1].set_title('Color Intensity Map')
        self.axes[1,0].set_title('Scale Bar - Draw line from start to end')
        self.axes[1,1].set_title('Path Results')
        
        # Reset button states
        self.btn_draw_line.ax.set_facecolor('lightgray')
        self.btn_find_paths.ax.set_facecolor('lightgray')
        
        # Redraw images if they exist
        if self.temporal_image is not None:
            self.axes[0,0].imshow(self.temporal_image)
            self.create_intensity_map()
        if self.scale_bar_image is not None:
            self.axes[1,0].imshow(self.scale_bar_image)
            self.btn_draw_line.ax.set_facecolor('lightgreen')
        
        plt.draw()
        self.update_status("All data cleared. Ready to start over.")
    
    def run(self):
        """Run the application"""
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    tracer = TemporalPathTracer()

    tracer.run()
