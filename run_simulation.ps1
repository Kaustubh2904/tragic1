# Crowd Simulation Runner Script
# Quick start script for Windows PowerShell

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Crowd Simulation & Evacuation System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python not found. Please install Python 3.7 or higher." -ForegroundColor Red
    exit 1
}

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$pipList = pip list 2>$null
if ($pipList -notmatch "numpy") {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

Write-Host "Dependencies OK" -ForegroundColor Green
Write-Host ""

# Menu
Write-Host "Select a scenario to run:" -ForegroundColor Cyan
Write-Host "1. Default Configuration (500 agents)"
Write-Host "2. Railway Station (300 agents, fire hazard)"
Write-Host "3. School Building (400 agents, multiple fires)"
Write-Host "4. Custom (specify parameters)"
Write-Host ""

$choice = Read-Host "Enter choice (1-4)"

switch ($choice) {
    "1" {
        Write-Host "Running default configuration..." -ForegroundColor Green
        python main.py
    }
    "2" {
        Write-Host "Running railway station scenario..." -ForegroundColor Green
        python main.py --config scenarios/railway_station.yaml
    }
    "3" {
        Write-Host "Running school building scenario..." -ForegroundColor Green
        python main.py --config scenarios/school_building.yaml
    }
    "4" {
        $agents = Read-Host "Number of agents (default: 500)"
        $duration = Read-Host "Duration in seconds (default: 300)"
        $noViz = Read-Host "Disable visualization? (y/n)"
        
        $cmd = "python main.py"
        if ($agents) { $cmd += " --agents $agents" }
        if ($duration) { $cmd += " --duration $duration" }
        if ($noViz -eq "y") { $cmd += " --no-viz" }
        
        Write-Host "Running custom simulation..." -ForegroundColor Green
        Invoke-Expression $cmd
    }
    default {
        Write-Host "Invalid choice. Running default configuration..." -ForegroundColor Yellow
        python main.py
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Simulation complete!" -ForegroundColor Green
Write-Host "Check the 'output' folder for results." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
