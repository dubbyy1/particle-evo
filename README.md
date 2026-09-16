# particle-evo
<img width="400" height="400" alt="moon_with_ui" src="https://raw.githubusercontent.com/dubbyy1/particle-evo/refs/heads/main/img/demo.gif" />
A Particle Life simulation using genomes.

## What's a Genome?
In traditional Particle Life particles are divided into colors, and their relationships are defined by a square matrix. My version uses a fixed number of species, each with their own genome influencing how they interact.

A genome is made of **3 traits** (0 to 1) and **3 receptors** (-1 to 1) and each receptor interacts with it’s respective trait on a neighbouring particle. For example, if Trait 1 on Particle _A_ is **1**, and Receptor 2 on Particle _B_ is also **1**, _B_ will be _attracted_ to _A_. Likewise, if Receptor 1 on Particle _B_ is **-1**, _B_ will be _repelled_ by _A_.

## Features
 - GPU Acceleration and rendering
- Supports up to 100,000 particles (limit can be changed in the code)
- Up to 10 species
- Highly configurable
  - Modify individual genes
  - Adjust physics
  - Adjust visuals
 - Save and load configurations

## How to run
Download the latest relevant executable from the [releases page]. Press SPACE to toggle UI, and ESCAPE to close. Mouse to interact with buttons and sliders.

## System Requirements
Will run on a CPU but good GPU is required for higher particle counts. My GPU (RX 6800) can comfortably run 50,000+, whereas my CPU (7600X) caps out at ~8000 particles.

## Build instructions

### Prerequisites
- Python 3.12

Run the following commands inside the project folder:

Windows:
```powershell
python -m venv .venv	
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m PyInstaller ".\Particle Evolution.spec"
```
> If `.\.venv\Scripts\Activate.ps1` fails, try running `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pyinstaller 'Particle Evolution.spec'
chmod +x 'dist/Particle Evolution'
```

The compiled executable can be found at `dist/Particle Evolution`

> inspired by [Particle Life](https://sandbox-science.com/particle-life) by [DicSo92](https://github.com/DicSo92)
