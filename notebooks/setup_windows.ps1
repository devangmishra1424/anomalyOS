# Windows Setup Script for AnomalyOS
[CmdletBinding()]
param(
    [string]$Action = 'install'
)

function Install-Requirements {
    Write-Host "Installing Python packages..." -ForegroundColor Green
    
    # Check if venv exists
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
    }
    
    # Activate venv
    & ".\venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    
    # Install requirements
    Write-Host "Installing project dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    Write-Host "Installation complete!" -ForegroundColor Green
}

function Start-Development {
    Write-Host "Starting development environment..." -ForegroundColor Green
    
    # Activate venv
    & ".\venv\Scripts\Activate.ps1"
    
    Write-Host "Development environment ready!" -ForegroundColor Green
    Write-Host "Run 'python app.py' to start the API" -ForegroundColor Cyan
}

function Main {
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host "AnomalyOS Setup (Windows)" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host
    
    switch ($Action.ToLower()) {
        'install' {
            Install-Requirements
        }
        'dev' {
            Start-Development
        }
        default {
            Write-Host "Usage: .\setup_windows.ps1 -Action [install|dev]" -ForegroundColor Yellow
        }
    }
}

Main
