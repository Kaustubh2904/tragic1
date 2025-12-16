"""
Sample Floorplan Generator
Creates a simple test floorplan image for demonstration
"""

try:
    from PIL import Image, ImageDraw
    import numpy as np
except ImportError:
    print("Error: Requires Pillow. Install with: pip install Pillow")
    exit(1)

def create_sample_floorplan(output_path="floorplans/sample_office.png"):
    """Create a sample office floorplan image."""
    
    # Create 500x500 image (white background = walkable)
    width, height = 500, 500
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw outer walls (black)
    wall_thickness = 20
    # Top wall
    draw.rectangle([0, 0, width, wall_thickness], fill='black')
    # Bottom wall
    draw.rectangle([0, height-wall_thickness, width, height], fill='black')
    # Left wall
    draw.rectangle([0, 0, wall_thickness, height], fill='black')
    # Right wall
    draw.rectangle([width-wall_thickness, 0, width, height], fill='black')
    
    # Draw interior walls/rooms (black)
    # Vertical wall in middle
    draw.rectangle([width//2-10, 100, width//2+10, height-100], fill='black')
    
    # Horizontal walls
    draw.rectangle([100, height//2-10, width//2-50, height//2+10], fill='black')
    draw.rectangle([width//2+50, height//2-10, width-100, height//2+10], fill='black')
    
    # Draw obstacles (blue) - furniture, columns
    # Conference room table
    draw.rectangle([150, 150, 220, 220], fill='blue')
    
    # Office desks
    draw.rectangle([300, 150, 350, 180], fill='blue')
    draw.rectangle([300, 200, 350, 230], fill='blue')
    
    # Central pillar
    draw.ellipse([width//2-15, height//2-15, width//2+15, height//2+15], fill='blue')
    
    # Draw exits (red)
    # Exit 1 - Left side
    draw.rectangle([wall_thickness, height//2-30, wall_thickness+30, height//2+30], fill='red')
    
    # Exit 2 - Right side
    draw.rectangle([width-wall_thickness-30, height//2-30, width-wall_thickness, height//2+30], fill='red')
    
    # Exit 3 - Top
    draw.rectangle([width//2-30, wall_thickness, width//2+30, wall_thickness+30], fill='red')
    
    # Exit 4 - Bottom
    draw.rectangle([width//2-30, height-wall_thickness-30, width//2+30, height-wall_thickness], fill='red')
    
    # Save image
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path)
    
    print(f"✓ Sample floorplan created: {output_path}")
    print(f"  Size: {width}×{height} pixels")
    print(f"  Suggested scale: 10 pixels/meter (creates 50×50m building)")
    print(f"\nTo use:")
    print(f"  python main.py --floorplan {output_path} --scale 10")
    
    return output_path


def create_simple_floorplan(output_path="floorplans/simple_room.png"):
    """Create a very simple single-room floorplan."""
    
    # Create 400x300 image
    width, height = 400, 300
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw walls
    wall = 15
    draw.rectangle([0, 0, width, wall], fill='black')
    draw.rectangle([0, height-wall, width, height], fill='black')
    draw.rectangle([0, 0, wall, height], fill='black')
    draw.rectangle([width-wall, 0, width, height], fill='black')
    
    # Central obstacle
    draw.rectangle([width//2-40, height//2-30, width//2+40, height//2+30], fill='blue')
    
    # Two exits
    draw.rectangle([wall, height//2-25, wall+25, height//2+25], fill='red')
    draw.rectangle([width-wall-25, height//2-25, width-wall, height//2+25], fill='red')
    
    # Save
    import os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path)
    
    print(f"✓ Simple floorplan created: {output_path}")
    print(f"  Size: {width}×{height} pixels")
    print(f"  Suggested scale: 10 pixels/meter (creates 40×30m room)")
    
    return output_path


if __name__ == '__main__':
    print("Creating sample floorplans...\n")
    create_sample_office()
    print()
    create_simple_floorplan()
    print("\nDone! Use these floorplans to test the simulation.")
