# Crowd Simulation & Evacuation System

A sophisticated crowd simulation and evacuation modeling system with AI-powered floorplan analysis, multiple motion models, and real-time hazard simulation.

## 🎯 Features

- **Multiple Motion Models**: Social Force Model (SFM), Reciprocal Velocity Obstacles (RVO), A* Pathfinding, and Hybrid approaches
- **AI-Powered Floorplan Analysis**: YOLOv8-based automatic detection of walls, doors, windows, and exits from architectural drawings
- **Real-time Hazard Simulation**: Dynamic fire spread, smoke propagation, and structural failures
- **Panic Dynamics**: Realistic panic behavior modeling and crowd psychology
- **Advanced Analytics**: Real-time metrics tracking, heatmap generation, and evacuation efficiency analysis
- **Rich Visualization**: Real-time simulation rendering with customizable views
- **Flexible Input**: Support for DXF, PNG, JPG, and manual configuration

## 📋 Requirements

- Python 3.7 or higher
- Windows, Linux, or macOS

## 🚀 Quick Start

### Installation

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running Simulations

#### Method 1: Using PowerShell Script (Windows)
```powershell
.\run_simulation.ps1
```

#### Method 2: Using Python Directly
```bash
# Basic simulation with default config
python main.py --config config.yaml

# With floorplan analysis
python main.py --config config.yaml --floorplan floorplans/office.png

# With custom settings
python main.py --config config.yaml --agents 1000 --duration 600

# Using YOLOv8 floorplan detection
python main.py --config config.yaml --floorplan floorplans/building.png --use-yolo
```

## 📁 Project Structure

```
tragic/
├── main.py                      # Main entry point and CLI
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── run_simulation.ps1          # PowerShell quick-start script
├── yolov8n.pt                  # YOLOv8 model weights
│
├── src/                        # Core simulation modules
│   ├── __init__.py
│   ├── simulation_engine.py    # Main simulation loop
│   ├── agent.py                # Agent behavior and state
│   ├── environment.py          # Environment, obstacles, exits
│   ├── motion_models.py        # SFM, RVO, pathfinding
│   ├── hazard_manager.py       # Fire, smoke, structural hazards
│   ├── analytics.py            # Metrics and data collection
│   ├── visualizer.py           # Real-time visualization
│   └── floorplan_parser.py     # Floorplan processing
│
├── analyze_floorplan.py        # Basic floorplan analyzer
├── analyze_floorplan_yolo.py   # AI-powered floorplan analyzer
├── create_sample_floorplan.py  # Sample floorplan generator
├── upload_floorplan.py         # Floorplan upload utility
│
└── output/                     # Generated results
    ├── floorplan_analytics.csv
    └── heatmaps/
```

## 🎮 Usage Examples

### 1. Create a Sample Floorplan
```bash
python create_sample_floorplan.py
```
This creates a sample office floorplan in `floorplans/sample_office.png`

### 2. Analyze a Floorplan
```bash
# Basic analysis
python analyze_floorplan.py floorplans/office.png

# AI-powered analysis with YOLOv8
python analyze_floorplan_yolo.py floorplans/office.png
```

### 3. Run a Simulation
```bash
# Default simulation
python main.py --config config.yaml

# With custom agent count
python main.py --config config.yaml --agents 500

# With floorplan and visualization
python main.py --config config.yaml --floorplan floorplans/office.png --visualize
```

### 4. Advanced Options
```bash
python main.py --config config.yaml \
    --floorplan floorplans/building.png \
    --agents 1000 \
    --duration 600 \
    --motion-model hybrid \
    --use-yolo \
    --visualize \
    --output-dir results/
```

## ⚙️ Configuration

Edit `config.yaml` to customize simulation parameters:

### Simulation Settings
```yaml
simulation:
  duration: 300.0      # Total simulation time (seconds)
  time_step: 0.1       # Time step size (seconds)
  seed: 42             # Random seed for reproducibility
```

### Environment
```yaml
environment:
  width: 50.0          # Environment width (meters)
  height: 50.0         # Environment height (meters)
  grid_resolution: 0.5 # Grid cell size (meters)
```

### Agents
```yaml
agents:
  count: 500
  speed_range: [0.8, 1.8]      # Min/max speed (m/s)
  radius_range: [0.2, 0.4]     # Agent radius (meters)
  visibility_range: [3.0, 15.0]
  panic_threshold: 0.3
  panic_spread_radius: 5.0
```

### Motion Models
```yaml
motion:
  model: "hybrid"  # Options: "sfm", "rvo", "pathfinding", "hybrid"
  sfm:
    relaxation_time: 0.5
    desired_speed_factor: 1.0
    agent_strength: 2000.0
  rvo:
    time_horizon: 2.0
    neighbor_dist: 5.0
  pathfinding:
    algorithm: "astar"
    replan_interval: 1.0
```

### Hazards
```yaml
hazards:
  fire:
    enabled: true
    start_time: 30.0
    ignition_points: [[25.0, 25.0]]
    spread_rate: 0.05
    damage_rate: 0.1
```

## 📊 Output and Analytics

The simulation generates several outputs:

- **Real-time Visualization**: Live animation of the simulation
- **Heatmaps**: Density and congestion heatmaps in `output/heatmaps/`
- **Analytics CSV**: Detailed metrics in `output/floorplan_analytics.csv`
- **Evacuation Metrics**: Evacuation times, bottleneck analysis, casualty reports

### Key Metrics Tracked:
- Evacuation completion time
- Average evacuation time per agent
- Bottleneck locations and severity
- Panic spread patterns
- Casualty counts and causes
- Exit utilization efficiency

## 🤖 AI-Powered Floorplan Analysis

The system uses YOLOv8 for automatic floorplan detection:

### Detectable Elements:
- **Walls**: Structural boundaries
- **Doors**: Entry/exit points
- **Windows**: Potential emergency exits
- **Exits**: Main evacuation points
- **Stairs**: Vertical circulation
- **Elevators**: (Not used in evacuation)

### Usage:
```bash
# Train or use existing YOLOv8 model
python analyze_floorplan_yolo.py floorplans/building.png --model yolov8n.pt
```

## 🔬 Motion Models

### Social Force Model (SFM)
- Physics-based crowd simulation
- Models attractive/repulsive forces
- Best for: Dense crowds, panic scenarios

### Reciprocal Velocity Obstacles (RVO)
- Collision avoidance optimization
- Distributed control
- Best for: Moderate crowds, orderly movement

### A* Pathfinding
- Optimal path planning
- Grid-based navigation
- Best for: Complex environments, obstacle avoidance

### Hybrid Model
- Combines strengths of all models
- Adaptive based on crowd density
- Best for: General-purpose simulations

## 🛠️ Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_simulation.py
```

### Adding Custom Motion Models
1. Extend the `MotionController` class in `src/motion_models.py`
2. Implement the `compute_desired_velocity()` method
3. Register in the configuration

### Adding Custom Hazards
1. Extend the `HazardManager` class in `src/hazard_manager.py`
2. Implement spread and effect logic
3. Configure in `config.yaml`

## 📝 Command-Line Arguments

```
usage: main.py [-h] --config CONFIG [--floorplan FLOORPLAN] 
               [--agents AGENTS] [--duration DURATION]
               [--motion-model {sfm,rvo,pathfinding,hybrid}]
               [--use-yolo] [--visualize] [--output-dir OUTPUT_DIR]

Arguments:
  --config CONFIG           Path to configuration YAML file
  --floorplan FLOORPLAN     Path to floorplan image or DXF file
  --agents AGENTS           Number of agents to simulate
  --duration DURATION       Simulation duration in seconds
  --motion-model MODEL      Motion model to use
  --use-yolo                Use YOLOv8 for floorplan detection
  --visualize               Enable real-time visualization
  --output-dir DIR          Output directory for results
```

## 📦 Dependencies

Core dependencies:
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `opencv-python` - Image processing
- `pillow` - Image handling
- `pyyaml` - Configuration parsing
- `scipy` - Scientific computing
- `ultralytics` - YOLOv8 for AI detection
- `torch` - Deep learning backend
- `ezdxf` - DXF file parsing
- `polars` - Fast data processing
- `psutil` - System monitoring

See `requirements.txt` for complete list.

## 🎯 Use Cases

- **Emergency Evacuation Planning**: Test evacuation procedures for buildings
- **Crowd Management**: Design safer venues and events
- **Architecture & Design**: Optimize building layouts for safety
- **Training & Education**: Demonstrate crowd dynamics and safety principles
- **Research**: Study crowd behavior, panic dynamics, and evacuation strategies

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional motion models
- More hazard types (gas leaks, flooding, etc.)
- Machine learning for behavior prediction
- 3D visualization
- Multi-floor buildings
- Real-time simulation control

## 📄 License

This project is provided as-is for educational and research purposes.

## 🐛 Troubleshooting

### Common Issues:

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**YOLOv8 model not found:**
- Download from: https://github.com/ultralytics/ultralytics
- Place `yolov8n.pt` in project root

**Visualization not appearing:**
- Check matplotlib backend: `python -c "import matplotlib; print(matplotlib.get_backend())"`
- Try: `export MPLBACKEND=TkAgg` (Linux/Mac) or set in code

**Memory issues with large agent counts:**
- Reduce agent count in config
- Disable visualization for large simulations
- Increase time_step to reduce computation

## 📧 Support

For issues, questions, or contributions, please refer to the project documentation or contact the development team.

## 🚀 Roadmap

- [ ] Web-based interface
- [ ] Real-time parameter adjustment
- [ ] Multi-floor support
- [ ] VR/AR visualization
- [ ] Machine learning-based behavior prediction
- [ ] Integration with BIM (Building Information Modeling)
- [ ] Mobile app for on-site simulations

---

**Version**: 1.0.0  
**Last Updated**: December 2025
