"""
Floorplan Analyzer - Extract Walls and Exits
Takes a user floor plan as input and outputs a simplified PNG showing only walls and exits
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle, Circle

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


def detect_file_type(filepath: str) -> str:
    """Detect floorplan file type from extension."""
    ext = Path(filepath).suffix.lower()
    
    if ext == '.dxf':
        if not EZDXF_AVAILABLE:
            print("Error: ezdxf is required for DXF files. Install with: pip install ezdxf")
            sys.exit(1)
        return 'dxf'
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        return 'image'
    else:
        print(f"Error: Unsupported file type '{ext}'. Supported: .png, .jpg, .jpeg, .bmp, .dxf")
        sys.exit(1)


def analyze_image_floorplan(filepath: str, scale: float = 10.0):
    """
    Analyze an image floorplan and extract walls and exits.
    Uses adaptive thresholding to work with any image.
    
    Args:
        filepath: Path to image file
        scale: Pixels per meter (default: 10)
    
    Returns:
        Tuple of (walls, exits, width, height)
    """
    print(f"\n📊 Analyzing image floorplan: {filepath}")
    
    # Load image
    img = Image.open(filepath)
    image_array = np.array(img.convert('RGB'))
    
    height_px, width_px = image_array.shape[:2]
    print(f"  Image size: {width_px}×{height_px} pixels")
    
    # Calculate world dimensions
    width_m = width_px / scale
    height_m = height_px / scale
    print(f"  World size: {width_m:.1f}×{height_m:.1f} meters (scale: {scale} px/m)")
    
    # Convert to grayscale
    grayscale = np.mean(image_array, axis=2)
    
    # Adaptive thresholding using Otsu's method
    hist, _ = np.histogram(grayscale, bins=256, range=(0, 256))
    
    total = grayscale.size
    sum_total = np.sum(np.arange(256) * hist)
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = i
    
    print(f"  Auto-detected threshold: {threshold:.0f}/255")
    
    # Create wall mask (dark areas)
    wall_mask = grayscale < threshold
    
    wall_percent = 100 * np.sum(wall_mask) / wall_mask.size
    print(f"  Wall coverage: {wall_percent:.1f}%")
    
    # Extract wall regions using contour detection
    walls = extract_wall_regions(wall_mask, scale)
    print(f"  Extracted {len(walls)} wall segments")
    
    # Detect exits (look for red/green regions or edge openings)
    exits = detect_exits_from_image(image_array, wall_mask, scale)
    print(f"  Detected {len(exits)} exits")
    
    return walls, exits, width_m, height_m


def extract_wall_regions(wall_mask, scale):
    """Extract wall rectangles from binary mask."""
    walls = []
    
    # Downsample for performance
    stride = max(1, int(scale / 2))
    
    height, width = wall_mask.shape
    
    # Simple grid-based extraction
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if wall_mask[y, x]:
                # Find extent of wall region
                x_end = x
                while x_end < width and wall_mask[y, x_end]:
                    x_end += 1
                
                y_end = y
                while y_end < height and wall_mask[y_end, x]:
                    y_end += 1
                
                # Convert to world coordinates
                wall_x = x / scale
                wall_y = y / scale
                wall_w = (x_end - x) / scale
                wall_h = (y_end - y) / scale
                
                if wall_w > 0.1 and wall_h > 0.1:  # Minimum size threshold
                    walls.append({
                        'x': wall_x,
                        'y': wall_y,
                        'width': wall_w,
                        'height': wall_h
                    })
    
    # Merge overlapping rectangles
    walls = merge_rectangles(walls)
    
    return walls


def merge_rectangles(rectangles, threshold=0.5):
    """Merge overlapping or nearby rectangles."""
    if not rectangles:
        return []
    
    merged = []
    used = set()
    
    for i, rect1 in enumerate(rectangles):
        if i in used:
            continue
        
        current = rect1.copy()
        used.add(i)
        
        # Try to merge with others
        merged_any = True
        while merged_any:
            merged_any = False
            for j, rect2 in enumerate(rectangles):
                if j in used or j == i:
                    continue
                
                # Check if rectangles overlap or are close
                if rectangles_overlap(current, rect2, threshold):
                    # Merge them
                    min_x = min(current['x'], rect2['x'])
                    min_y = min(current['y'], rect2['y'])
                    max_x = max(current['x'] + current['width'], rect2['x'] + rect2['width'])
                    max_y = max(current['y'] + current['height'], rect2['y'] + rect2['height'])
                    
                    current = {
                        'x': min_x,
                        'y': min_y,
                        'width': max_x - min_x,
                        'height': max_y - min_y
                    }
                    used.add(j)
                    merged_any = True
        
        merged.append(current)
    
    return merged


def rectangles_overlap(rect1, rect2, threshold=0.5):
    """Check if two rectangles overlap or are within threshold distance."""
    r1_right = rect1['x'] + rect1['width']
    r1_bottom = rect1['y'] + rect1['height']
    r2_right = rect2['x'] + rect2['width']
    r2_bottom = rect2['y'] + rect2['height']
    
    # Check overlap with threshold
    return not (r1_right + threshold < rect2['x'] or 
                rect1['x'] > r2_right + threshold or
                r1_bottom + threshold < rect2['y'] or
                rect1['y'] > r2_bottom + threshold)


def detect_exits_from_image(image_array, wall_mask, scale):
    """Detect exits from image using color detection."""
    exits = []
    
    # Look for red or green regions (common for exits/doors)
    red = image_array[:, :, 0].astype(float)
    green = image_array[:, :, 1].astype(float)
    blue = image_array[:, :, 2].astype(float)
    
    # Detect red areas (exits often marked in red)
    red_mask = (red > green + 30) & (red > blue + 30) & (red > 100)
    
    # Detect green areas (exits sometimes marked in green)
    green_mask = (green > red + 30) & (green > blue + 30) & (green > 100)
    
    exit_mask = red_mask | green_mask
    
    # Find exit locations
    if np.any(exit_mask):
        # Find connected components
        labeled = label_regions(exit_mask)
        
        for label_id in range(1, labeled.max() + 1):
            region = labeled == label_id
            coords = np.argwhere(region)
            
            if len(coords) > 20:  # Minimum size
                center_y = np.mean(coords[:, 0])
                center_x = np.mean(coords[:, 1])
                
                exits.append({
                    'x': center_x / scale,
                    'y': center_y / scale,
                    'width': 2.0  # Default exit width
                })
    
    # If no colored exits found, detect from edges
    if not exits:
        exits = detect_exits_from_edges(wall_mask, scale)
    
    return exits


def label_regions(binary_mask):
    """Simple connected component labeling."""
    try:
        from scipy import ndimage
        labeled, _ = ndimage.label(binary_mask)
        return labeled
    except ImportError:
        # Fallback: simple flood-fill based labeling
        labeled = np.zeros_like(binary_mask, dtype=int)
        label_id = 0
        
        def flood_fill(y, x, label):
            if y < 0 or y >= binary_mask.shape[0] or x < 0 or x >= binary_mask.shape[1]:
                return
            if not binary_mask[y, x] or labeled[y, x] != 0:
                return
            
            labeled[y, x] = label
            flood_fill(y-1, x, label)
            flood_fill(y+1, x, label)
            flood_fill(y, x-1, label)
            flood_fill(y, x+1, label)
        
        for y in range(binary_mask.shape[0]):
            for x in range(binary_mask.shape[1]):
                if binary_mask[y, x] and labeled[y, x] == 0:
                    label_id += 1
                    flood_fill(y, x, label_id)
        
        return labeled


def detect_exits_from_edges(wall_mask, scale):
    """Detect potential exits at the edges of the floor plan."""
    exits = []
    height, width = wall_mask.shape
    
    # Check all four edges for openings
    edges = [
        ('top', 0, range(0, width, int(scale * 2))),
        ('bottom', height - 1, range(0, width, int(scale * 2))),
        ('left', range(0, height, int(scale * 2)), 0),
        ('right', range(0, height, int(scale * 2)), width - 1)
    ]
    
    for edge_name, *coords in edges:
        if edge_name in ['top', 'bottom']:
            y, x_range = coords
            for x in x_range:
                if not wall_mask[y, x]:
                    exits.append({
                        'x': x / scale,
                        'y': y / scale,
                        'width': 2.0
                    })
        else:
            y_range, x = coords
            for y in y_range:
                if not wall_mask[y, x]:
                    exits.append({
                        'x': x / scale,
                        'y': y / scale,
                        'width': 2.0
                    })
    
    return exits[:4]  # Limit to 4 exits


def analyze_dxf_floorplan(filepath: str, scale: float = 100.0):
    """
    Analyze a DXF floorplan and extract walls and exits.
    
    Args:
        filepath: Path to DXF file
        scale: DXF units per meter (default: 100 for cm)
    
    Returns:
        Tuple of (walls, exits, width, height)
    """
    print(f"\n📊 Analyzing DXF floorplan: {filepath}")
    
    doc = ezdxf.readfile(filepath)
    msp = doc.modelspace()
    
    walls = []
    exits = []
    
    all_points = []
    
    # Extract lines and polylines as walls
    for entity in msp.query('LINE'):
        start = [entity.dxf.start.x / scale, entity.dxf.start.y / scale]
        end = [entity.dxf.end.x / scale, entity.dxf.end.y / scale]
        
        all_points.extend([start, end])
        
        walls.append({
            'type': 'line',
            'start': start,
            'end': end
        })
    
    for entity in msp.query('LWPOLYLINE POLYLINE'):
        points = [[p[0] / scale, p[1] / scale] for p in entity.get_points()]
        all_points.extend(points)
        
        for i in range(len(points) - 1):
            walls.append({
                'type': 'line',
                'start': points[i],
                'end': points[i + 1]
            })
    
    # Extract circles as potential exits
    exit_layers = ['EXIT', 'EXITS', 'DOOR', 'DOORS']
    for entity in msp.query('CIRCLE'):
        layer = entity.dxf.layer.upper() if hasattr(entity.dxf, 'layer') else ''
        
        center = [entity.dxf.center.x / scale, entity.dxf.center.y / scale]
        radius = entity.dxf.radius / scale
        
        all_points.append(center)
        
        if any(name in layer for name in exit_layers):
            exits.append({
                'x': center[0],
                'y': center[1],
                'width': radius * 2
            })
    
    # Calculate bounds
    if all_points:
        all_points = np.array(all_points)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        
        width = max_x - min_x + 10  # Add padding
        height = max_y - min_y + 10
    else:
        width, height = 50, 50
    
    print(f"  Extracted {len(walls)} wall segments")
    print(f"  Detected {len(exits)} exits")
    print(f"  World size: {width:.1f}×{height:.1f} meters")
    
    return walls, exits, width, height


def create_simplified_png(walls, exits, width, height, output_path):
    """
    Create a simplified PNG showing only walls and exits.
    
    Args:
        walls: List of wall dictionaries
        exits: List of exit dictionaries
        width: World width in meters
        height: World height in meters
        output_path: Output PNG file path
    """
    print(f"\n🎨 Creating simplified PNG: {output_path}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax.set_title('Extracted Floor Plan - Walls and Exits', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Draw walls
    for wall in walls:
        if wall.get('type') == 'line':
            # Line segment
            start = wall['start']
            end = wall['end']
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   color='black', linewidth=3, solid_capstyle='round')
        else:
            # Rectangle
            rect = Rectangle(
                (wall['x'], wall['y']),
                wall['width'],
                wall['height'],
                facecolor='#333333',
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
    
    # Draw exits
    for i, exit_obj in enumerate(exits):
        circle = Circle(
            (exit_obj['x'], exit_obj['y']),
            exit_obj['width'] / 2,
            facecolor='#00FF00',
            edgecolor='darkgreen',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(circle)
        
        # Label
        ax.text(
            exit_obj['x'],
            exit_obj['y'],
            f'EXIT\n{i+1}',
            ha='center',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='darkgreen'
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#333333', edgecolor='black', label=f'Walls ({len(walls)})'),
        Patch(facecolor='#00FF00', edgecolor='darkgreen', label=f'Exits ({len(exits)})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add info text
    info_text = f"Dimensions: {width:.1f}m × {height:.1f}m\n"
    info_text += f"Walls: {len(walls)}\nExits: {len(exits)}"
    
    ax.text(
        0.02, 0.98,
        info_text,
        transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
        fontsize=10,
        fontfamily='monospace'
    )
    
    # Save
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved simplified floor plan to: {output_path}")


def main():
    """Main function to handle user input and process floor plan."""
    
    print("=" * 60)
    print("  FLOORPLAN ANALYZER")
    print("  Extract Walls and Exits from Floor Plans")
    print("=" * 60)
    
    # Get input file from user
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = input("\n📁 Enter path to floor plan file: ").strip()
    
    # Remove quotes if present
    input_file = input_file.strip('"').strip("'")
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"\n❌ Error: File not found: {input_file}")
        sys.exit(1)
    
    # Detect file type
    file_type = detect_file_type(input_file)
    
    # Get scale (optional)
    if len(sys.argv) > 2:
        scale = float(sys.argv[2])
    else:
        if file_type == 'image':
            scale_input = input("\n📏 Enter scale (pixels per meter, default=10): ").strip()
            scale = float(scale_input) if scale_input else 10.0
        else:
            scale_input = input("\n📏 Enter scale (DXF units per meter, default=100): ").strip()
            scale = float(scale_input) if scale_input else 100.0
    
    # Generate output filename
    input_path = Path(input_file)
    output_file = f"output/{input_path.stem}_simplified.png"
    
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        output_input = input(f"\n💾 Enter output file (default={output_file}): ").strip()
        if output_input:
            output_file = output_input
    
    # Analyze floor plan
    try:
        if file_type == 'image':
            walls, exits, width, height = analyze_image_floorplan(input_file, scale)
        else:
            walls, exits, width, height = analyze_dxf_floorplan(input_file, scale)
        
        # Create output PNG
        create_simplified_png(walls, exits, width, height, output_file)
        
        print(f"\n{'=' * 60}")
        print(f"✅ SUCCESS! Simplified floor plan created.")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"\n❌ Error processing floor plan: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
