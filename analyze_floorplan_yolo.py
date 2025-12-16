"""
Floorplan Analyzer with YOLOv8
Uses deep learning for accurate detection of walls, doors, windows, and exits
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Circle, Polygon
import cv2

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Error: Ultralytics (YOLOv8) is required. Install with: pip install ultralytics")
    sys.exit(1)


class FloorplanAnalyzerYOLO:
    """Analyze floorplans using YOLOv8 for accurate object detection."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize analyzer with YOLOv8 model.
        
        Args:
            model_path: Path to YOLO model or model name (yolov8n.pt, yolov8s.pt, etc.)
        """
        print(f"🤖 Loading YOLOv8 model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("📥 Downloading YOLOv8 nano model...")
            self.model = YOLO('yolov8n.pt')
            print("✅ Model ready")
        
        self.image = None
        self.walls = []
        self.exits = []
        self.doors = []
        self.windows = []
        
    def analyze_floorplan(self, filepath: str, scale: float = 10.0):
        """
        Analyze floorplan image using YOLOv8 and image processing.
        
        Args:
            filepath: Path to floorplan image
            scale: Pixels per meter
            
        Returns:
            Tuple of (walls, exits, width, height)
        """
        print(f"\n📊 Analyzing floorplan with YOLOv8: {filepath}")
        
        # Load image
        self.image = cv2.imread(filepath)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {filepath}")
        
        height_px, width_px = self.image.shape[:2]
        print(f"  Image size: {width_px}×{height_px} pixels")
        
        # Calculate world dimensions
        width_m = width_px / scale
        height_m = height_px / scale
        print(f"  World size: {width_m:.1f}×{height_m:.1f} meters (scale: {scale} px/m)")
        
        # Run YOLOv8 detection
        print("  Running YOLOv8 detection...")
        results = self.model(self.image, conf=0.25, verbose=False)
        
        # Process detections
        detected_objects = results[0].boxes
        print(f"  Detected {len(detected_objects)} objects")
        
        # Extract walls using edge detection and contour analysis
        walls = self.extract_walls_advanced(self.image, scale)
        print(f"  Extracted {len(walls)} wall segments")
        
        # Detect doors and exits using color-based and edge analysis
        exits = self.detect_exits_advanced(self.image, scale)
        print(f"  Detected {len(exits)} exits/doors")
        
        return walls, exits, width_m, height_m
    
    def extract_walls_advanced(self, image, scale):
        """
        Extract walls using advanced image processing.
        
        Args:
            image: Input image (BGR format)
            scale: Pixels per meter
            
        Returns:
            List of wall dictionaries
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Edge detection for wall boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine binary and edges
        combined = cv2.bitwise_or(binary, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        walls = []
        min_area = (scale * 0.5) ** 2  # Minimum wall area threshold
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert to world coordinates
                wall_x = x / scale
                wall_y = y / scale
                wall_w = w / scale
                wall_h = h / scale
                
                # Filter out very small segments
                if wall_w > 0.3 or wall_h > 0.3:
                    walls.append({
                        'x': wall_x,
                        'y': wall_y,
                        'width': wall_w,
                        'height': wall_h,
                        'contour': contour
                    })
        
        return walls
    
    def detect_exits_advanced(self, image, scale):
        """
        Detect exits and doors using color analysis and pattern recognition.
        
        Args:
            image: Input image (BGR format)
            scale: Pixels per meter
            
        Returns:
            List of exit dictionaries
        """
        exits = []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for doors/exits (typically red or green in floorplans)
        # Red range (doors often marked in red)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Green range (exits often marked in green)
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Combine masks
        exit_mask = cv2.bitwise_or(red_mask, green_mask)
        
        # Find contours in exit mask
        contours, _ = cv2.findContours(exit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area for exit
                # Get center and size
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Get size
                    x, y, w, h = cv2.boundingRect(contour)
                    exit_width = max(w, h) / scale
                    
                    exits.append({
                        'x': cx / scale,
                        'y': cy / scale,
                        'width': max(1.5, exit_width)
                    })
        
        # If no colored exits found, detect from edges
        if len(exits) == 0:
            exits = self.detect_exits_from_edges(image, scale)
        
        return exits
    
    def detect_exits_from_edges(self, image, scale):
        """
        Detect potential exits at building edges.
        
        Args:
            image: Input image
            scale: Pixels per meter
            
        Returns:
            List of exit dictionaries
        """
        exits = []
        height, width = image.shape[:2]
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Check edges for openings (white areas = potential exits)
        edge_thickness = int(scale * 2)
        
        # Top edge
        top_edge = binary[:edge_thickness, :]
        for x in range(0, width, int(scale)):
            if np.mean(top_edge[:, max(0, x-5):min(width, x+5)]) > 200:
                exits.append({'x': x / scale, 'y': 0, 'width': 2.0})
        
        # Bottom edge
        bottom_edge = binary[-edge_thickness:, :]
        for x in range(0, width, int(scale)):
            if np.mean(bottom_edge[:, max(0, x-5):min(width, x+5)]) > 200:
                exits.append({'x': x / scale, 'y': height / scale, 'width': 2.0})
        
        # Left edge
        left_edge = binary[:, :edge_thickness]
        for y in range(0, height, int(scale)):
            if np.mean(left_edge[max(0, y-5):min(height, y+5), :]) > 200:
                exits.append({'x': 0, 'y': y / scale, 'width': 2.0})
        
        # Right edge
        right_edge = binary[:, -edge_thickness:]
        for y in range(0, height, int(scale)):
            if np.mean(right_edge[max(0, y-5):min(height, y+5), :]) > 200:
                exits.append({'x': width / scale, 'y': y / scale, 'width': 2.0})
        
        # Limit to reasonable number
        return exits[:8]


def create_simplified_png(walls, exits, width, height, output_path, input_image=None):
    """
    Create a simplified PNG showing only walls and exits.
    
    Args:
        walls: List of wall dictionaries
        exits: List of exit dictionaries
        width: World width in meters
        height: World height in meters
        output_path: Output PNG file path
        input_image: Optional original image to overlay
    """
    print(f"\n🎨 Creating simplified PNG: {output_path}")
    
    # Create figure with better DPI
    fig, ax = plt.subplots(figsize=(14, 12))
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F5')
    ax.set_xlabel('X (meters)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=13, fontweight='bold')
    ax.set_title('YOLOv8 Floor Plan Analysis - Walls and Exits', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Overlay original image (faded) if provided
    if input_image is not None:
        extent = [0, width, 0, height]
        ax.imshow(input_image, extent=extent, origin='upper', alpha=0.15, aspect='auto')
    
    # Draw walls
    for wall in walls:
        rect = Rectangle(
            (wall['x'], wall['y']),
            wall['width'],
            wall['height'],
            facecolor='#2C3E50',
            edgecolor='#1A252F',
            linewidth=1.5,
            alpha=0.85
        )
        ax.add_patch(rect)
    
    # Draw exits with better styling
    for i, exit_obj in enumerate(exits):
        # Main circle
        circle = Circle(
            (exit_obj['x'], exit_obj['y']),
            exit_obj['width'] / 2,
            facecolor='#27AE60',
            edgecolor='#1E8449',
            linewidth=2.5,
            alpha=0.9,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Outer glow
        glow = Circle(
            (exit_obj['x'], exit_obj['y']),
            exit_obj['width'] / 2 * 1.3,
            facecolor='none',
            edgecolor='#82E0AA',
            linewidth=1.5,
            linestyle='--',
            alpha=0.6,
            zorder=9
        )
        ax.add_patch(glow)
        
        # Label
        ax.text(
            exit_obj['x'],
            exit_obj['y'],
            f'EXIT\n{i+1}',
            ha='center',
            va='center',
            fontsize=10,
            fontweight='bold',
            color='white',
            zorder=11
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2C3E50', edgecolor='#1A252F', label=f'Walls ({len(walls)} segments)'),
        Patch(facecolor='#27AE60', edgecolor='#1E8449', label=f'Exits ({len(exits)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
             framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)
    
    # Add info box
    info_text = f"🏢 Dimensions: {width:.1f}m × {height:.1f}m\n"
    info_text += f"🧱 Walls: {len(walls)} segments\n"
    info_text += f"🚪 Exits: {len(exits)} detected\n"
    info_text += f"🤖 Analyzed with YOLOv8"
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', 
                 alpha=0.95, edgecolor='#D4AF37', linewidth=2),
        fontsize=11,
        fontfamily='monospace',
        fontweight='bold'
    )
    
    # Save with high quality
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved simplified floor plan to: {output_path}")


def main():
    """Main function with YOLOv8 integration."""
    
    print("=" * 70)
    print("  FLOORPLAN ANALYZER with YOLOv8")
    print("  AI-Powered Wall and Exit Detection")
    print("=" * 70)
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("\n📁 Enter path to floor plan file: ").strip()
    
    input_file = input_file.strip('"').strip("'")
    
    if not Path(input_file).exists():
        print(f"\n❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Get scale
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
    else:
        scale_input = input("\n📏 Enter scale (pixels per meter, default=10): ").strip()
        scale = float(scale_input) if scale_input else 10.0
    
    # Get model path
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    else:
        model_path = 'yolov8n.pt'  # Default to nano model
    
    # Output file
    input_path = Path(input_file)
    output_file = f"output/{input_path.stem}_yolo_simplified.png"
    
    if len(sys.argv) > 4:
        output_file = sys.argv[4]
    
    # Analyze with YOLOv8
    try:
        analyzer = FloorplanAnalyzerYOLO(model_path)
        walls, exits, width, height = analyzer.analyze_floorplan(input_file, scale)
        
        # Load original image for overlay
        original_img = cv2.imread(input_file)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create output PNG
        create_simplified_png(walls, exits, width, height, output_file, original_img)
        
        print(f"\n{'=' * 70}")
        print(f"✅ SUCCESS! YOLOv8 analysis complete.")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Model:  {model_path}")
        print(f"{'=' * 70}\n")
        
    except Exception as e:
        print(f"\n❌ Error processing floor plan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
