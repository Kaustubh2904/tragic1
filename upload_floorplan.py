"""
Interactive Floorplan Upload and Configuration
Allows users to upload floorplans and automatically configure simulation
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False


class FloorplanConfigurator:
    """Interactive floorplan upload and auto-configuration."""
    
    def __init__(self):
        self.floorplan_path = None
        self.floorplan_type = None
        self.scale = 1.0
        self.width = 50.0
        self.height = 50.0
        self.exits = []
        self.obstacles = []
    
    def detect_file_type(self, filepath: str) -> Optional[str]:
        """Detect floorplan file type from extension."""
        ext = Path(filepath).suffix.lower()
        
        if ext == '.dxf':
            if not EZDXF_AVAILABLE:
                print("⚠️  Warning: ezdxf not installed. Install with: pip install ezdxf")
                return None
            return 'dxf'
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            if not PIL_AVAILABLE:
                print("⚠️  Warning: Pillow not installed. Install with: pip install Pillow")
                return None
            return 'image'
        else:
            print(f"❌ Unsupported file type: {ext}")
            return None
    
    def analyze_image_floorplan(self, filepath: str):
        """Analyze image floorplan and extract parameters."""
        img = Image.open(filepath)
        width_px, height_px = img.shape if hasattr(img, 'shape') else img.size
        
        print(f"\n📐 Image dimensions: {width_px} × {height_px} pixels")
        
        # Auto-detect scale
        print("\n🔍 Analyzing floorplan...")
        
        # Convert to numpy array
        img_array = np.array(img.convert('RGB'))
        
        # Detect exits (red pixels)
        red_mask = (img_array[:, :, 0] > 200) & (img_array[:, :, 1] < 100) & (img_array[:, :, 2] < 100)
        exit_count = np.sum(red_mask) / (width_px * height_px)
        
        # Detect walkable area (light pixels)
        gray = np.mean(img_array, axis=2)
        walkable_area = np.sum(gray > 200) / (width_px * height_px)
        
        print(f"  • Walkable area: {walkable_area*100:.1f}%")
        print(f"  • Exit markers: {exit_count*100:.2f}% of image")
        
        return width_px, height_px, walkable_area
    
    def get_user_input(self, prompt: str, default: str = "") -> str:
        """Get user input with default value."""
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            return user_input if user_input else default
        return input(f"{prompt}: ").strip()
    
    def configure_from_floorplan(self, filepath: str):
        """Interactively configure simulation from floorplan."""
        print("\n" + "="*60)
        print("🏢 FLOORPLAN CONFIGURATION WIZARD")
        print("="*60)
        
        # Verify file exists
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return None
        
        self.floorplan_path = filepath
        self.floorplan_type = self.detect_file_type(filepath)
        
        if not self.floorplan_type:
            return None
        
        print(f"✅ Detected floorplan type: {self.floorplan_type.upper()}")
        
        # Analyze floorplan
        if self.floorplan_type == 'image':
            width_px, height_px, walkable = self.analyze_image_floorplan(filepath)
            
            print("\n📏 SCALE CONFIGURATION")
            print("How large is the real building?")
            
            # Get real-world dimensions
            print("\nOption 1: Specify total dimensions")
            has_dimensions = self.get_user_input("Do you know the total building dimensions? (y/n)", "y")
            
            if has_dimensions.lower() == 'y':
                self.width = float(self.get_user_input("Building width in meters", "50"))
                self.height = float(self.get_user_input("Building height in meters", "50"))
                self.scale = max(width_px / self.width, height_px / self.height)
            else:
                print("\nOption 2: Specify scale")
                print("Example: If 1 pixel = 0.1 meters, enter 10 (pixels per meter)")
                self.scale = float(self.get_user_input("Pixels per meter", "10"))
                self.width = width_px / self.scale
                self.height = height_px / self.scale
            
            print(f"\n✅ Configured: {self.width:.1f}m × {self.height:.1f}m")
            print(f"   Scale: {self.scale:.2f} pixels/meter")
        
        elif self.floorplan_type == 'dxf':
            print("\n📏 DXF files use their internal units")
            self.scale = float(self.get_user_input("DXF units per meter (usually 100 for cm)", "100"))
            
            # Try to get bounds from DXF
            try:
                doc = ezdxf.readfile(filepath)
                auditor = doc.audit()
                
                # Get approximate dimensions
                print("   Analyzing DXF structure...")
                msp = doc.modelspace()
                
                # Count entities
                lines = len(list(msp.query('LINE')))
                circles = len(list(msp.query('CIRCLE')))
                
                print(f"   Found: {lines} lines, {circles} circles")
                
            except Exception as e:
                print(f"   ⚠️  Could not analyze DXF: {e}")
            
            self.width = float(self.get_user_input("Estimated width in meters", "50"))
            self.height = float(self.get_user_input("Estimated height in meters", "50"))
        
        # Agent configuration
        print("\n👥 AGENT CONFIGURATION")
        agent_count = int(self.get_user_input("Number of agents", "300"))
        
        # Estimate density
        area = self.width * self.height
        if self.floorplan_type == 'image':
            usable_area = area * walkable
        else:
            usable_area = area * 0.7  # Assume 70% walkable
        
        density = agent_count / usable_area
        print(f"   Estimated density: {density:.2f} agents/m²")
        
        if density > 5:
            print("   ⚠️  Warning: Very high density! Consider reducing agents.")
        elif density < 0.5:
            print("   ℹ️  Note: Low density - may evacuate quickly.")
        
        # Simulation parameters
        print("\n⏱️  SIMULATION PARAMETERS")
        duration = float(self.get_user_input("Simulation duration (seconds)", "300"))
        
        # Hazard configuration
        print("\n🔥 HAZARD CONFIGURATION")
        enable_fire = self.get_user_input("Enable fire? (y/n)", "y").lower() == 'y'
        
        fire_start_time = 30.0
        fire_locations = []
        
        if enable_fire:
            fire_start_time = float(self.get_user_input("Fire start time (seconds)", "30"))
            
            print("\nFire location options:")
            print("  1. Center of building")
            print("  2. Random location")
            print("  3. Custom coordinates")
            
            fire_option = self.get_user_input("Choose option (1-3)", "1")
            
            if fire_option == "1":
                fire_locations = [[self.width/2, self.height/2]]
            elif fire_option == "2":
                import random
                fire_locations = [[
                    random.uniform(self.width*0.2, self.width*0.8),
                    random.uniform(self.height*0.2, self.height*0.8)
                ]]
            else:
                x = float(self.get_user_input(f"Fire X coordinate (0-{self.width:.1f})", str(self.width/2)))
                y = float(self.get_user_input(f"Fire Y coordinate (0-{self.height:.1f})", str(self.height/2)))
                fire_locations = [[x, y]]
        
        # Exit detection
        print("\n🚪 EXIT CONFIGURATION")
        
        if self.floorplan_type == 'image':
            auto_detect = self.get_user_input("Auto-detect exits from red markers? (y/n)", "y").lower() == 'y'
            
            if auto_detect:
                print("   Using red pixels in image as exit markers...")
            else:
                print("   Exits will be placed at building corners")
        else:
            print("   DXF exits will be detected from CIRCLE entities on EXIT layer")
        
        # Generate configuration
        config = self.generate_config(
            agent_count=agent_count,
            duration=duration,
            enable_fire=enable_fire,
            fire_start_time=fire_start_time,
            fire_locations=fire_locations
        )
        
        return config
    
    def generate_config(self, agent_count: int, duration: float, 
                       enable_fire: bool, fire_start_time: float,
                       fire_locations: List[List[float]]) -> dict:
        """Generate simulation configuration dictionary."""
        
        config = {
            'simulation': {
                'duration': duration,
                'time_step': 0.1,
                'seed': np.random.randint(0, 10000)
            },
            'environment': {
                'width': self.width,
                'height': self.height,
                'grid_resolution': 0.5
            },
            'agents': {
                'count': agent_count,
                'speed_range': [1.0, 1.6],
                'radius_range': [0.25, 0.35],
                'visibility_range': [5.0, 12.0],
                'panic_threshold': 0.3,
                'panic_spread_radius': 5.0,
                'panic_spread_rate': 0.15,
                'max_panic_level': 1.0
            },
            'motion': {
                'model': 'hybrid',
                'sfm': {
                    'relaxation_time': 0.5,
                    'desired_speed_factor': 1.0,
                    'agent_strength': 2000.0,
                    'agent_range': 0.08,
                    'wall_strength': 2000.0,
                    'wall_range': 0.08,
                    'noise_factor': 0.1
                },
                'rvo': {
                    'time_horizon': 2.0,
                    'neighbor_dist': 5.0,
                    'max_neighbors': 10
                },
                'pathfinding': {
                    'algorithm': 'astar',
                    'replan_interval': 1.5,
                    'congestion_weight': 0.4,
                    'hazard_weight': 0.6
                }
            },
            'hazards': {
                'fire': {
                    'enabled': enable_fire,
                    'start_time': fire_start_time,
                    'ignition_points': fire_locations,
                    'spread_rate': 0.08,
                    'damage_rate': 0.15,
                    'growth_rate': 2.0
                },
                'smoke': {
                    'enabled': enable_fire,
                    'diffusion_rate': 0.4,
                    'visibility_reduction': 0.85,
                    'damage_rate': 0.03
                },
                'exit_failures': {
                    'enabled': False,
                    'failure_times': [],
                    'failure_exits': []
                }
            },
            'exits': {
                'positions': [
                    [self.width * 0.1, self.height * 0.5],
                    [self.width * 0.9, self.height * 0.5],
                    [self.width * 0.5, self.height * 0.1],
                    [self.width * 0.5, self.height * 0.9]
                ],
                'widths': [2.5, 2.5, 2.5, 2.5],
                'capacities': [200, 200, 200, 200]
            },
            'obstacles': {
                'rectangles': []
            },
            'floorplan': {
                'type': self.floorplan_type,
                'file_path': str(self.floorplan_path),
                'scale': self.scale,
                'origin': [0, 0]
            },
            'visualization': {
                'enabled': True,
                'fps': 30,
                'realtime': False,
                'show_trajectories': False,
                'show_panic_levels': True,
                'show_hazards': True,
                'video_export': False,
                'video_path': 'output/floorplan_simulation.mp4',
                'heatmap_export': True,
                'heatmap_path': 'output/heatmaps'
            },
            'analytics': {
                'enabled': True,
                'sampling_rate': 0.5,
                'export_csv': True,
                'csv_path': 'output/floorplan_analytics.csv',
                'compute_heatmaps': True,
                'bottleneck_threshold': 5.0
            },
            'output': {
                'directory': 'output',
                'save_final_state': True
            }
        }
        
        return config
    
    def save_config(self, config: dict, output_path: str = "scenarios/uploaded_floorplan.yaml"):
        """Save configuration to YAML file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n💾 Configuration saved to: {output_path}")
        return output_path


def main():
    """Main interactive floorplan upload interface."""
    
    print("\n" + "="*60)
    print("🏢 FLOORPLAN UPLOAD & SIMULATION CONFIGURATOR")
    print("="*60)
    print("\nSupported formats:")
    print("  • DXF (AutoCAD files) - requires: pip install ezdxf")
    print("  • Images (PNG, JPG) - requires: pip install Pillow")
    print("\nColor coding for images:")
    print("  • Black/Dark = Walls")
    print("  • White/Light = Walkable areas")
    print("  • Red = Exits")
    print("  • Blue = Obstacles")
    print("="*60)
    
    configurator = FloorplanConfigurator()
    
    # Get floorplan file
    print("\n📁 FLOORPLAN FILE")
    print("Enter the path to your floorplan file:")
    print("Example: floorplans/my_building.png")
    print("         C:/Users/YourName/Desktop/floor.dxf")
    
    filepath = input("\nFloorplan path: ").strip().strip('"').strip("'")
    
    if not filepath:
        print("❌ No file specified. Exiting.")
        return
    
    # Configure simulation
    config = configurator.configure_from_floorplan(filepath)
    
    if config is None:
        print("\n❌ Configuration failed.")
        return
    
    # Save configuration
    config_path = configurator.save_config(config)
    
    # Ask to run simulation
    print("\n" + "="*60)
    print("✅ CONFIGURATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"  • Building: {configurator.width:.1f}m × {configurator.height:.1f}m")
    print(f"  • Agents: {config['agents']['count']}")
    print(f"  • Duration: {config['simulation']['duration']:.0f}s")
    print(f"  • Fire: {'Yes' if config['hazards']['fire']['enabled'] else 'No'}")
    
    run_now = input("\n🚀 Run simulation now? (y/n) [y]: ").strip().lower()
    
    if run_now != 'n':
        print("\nStarting simulation...\n")
        os.system(f'python main.py --config {config_path}')
    else:
        print(f"\n💡 To run later, use:")
        print(f"   python main.py --config {config_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Configuration cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
