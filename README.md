# Soccer 3D Reconstruction
3D Reconstruction of soccer matches from multiview camera using `Gaussian Splatting`.

## Installation
Clone the repository.
```bash
git clone https://github.com/Kyaw-Thiha/soccer-recon.git
cd soccer-recon
```

Create the virtual environment.
```bash
conda create -n soccer-recon python=3.12 -y
```

Activate the virtual environment.
```bash
conda activate soccer-recon
```

Install GPU packages using conda.
```bash
conda install -c nvidia cuda-nvcc cuda-toolkit
```

Install the package.
```bash
pip install -e .
```

## Project Structure
```
soccer-recon/
├── configs/
│   ├── experiments/              
│   │   ├── baseline_gs.yaml
│   │   ├── multireso_gs.yaml
│   │   └── dynamic_gs.yaml
│   └── base.yaml         
│
├── soccer_recon/                  
│   │
│   ├── models/                   
│   │   ├── base_gs_model.py      
│   │   ├── multireso_model.py    
│   │   └── dynamic_model.py      
│   │
│   ├── fields/                   # NS Fields (3D representations)
│   │   ├── gaussian_field.py     
│   │   └── dynamic_field.py     
│   │
│   ├── methods/                  
│   │   ├── base_gs_method.py     
│   │   ├── multireso_method.py   
│   │   └── dynamic_method.py     
│   │
│   ├── data/
│   │   ├── dataparsers.py        
│   │   ├── datamanager.py        
│   │   └── datasets.py           
│   │
│   ├── pipelines/                
│   │   └── soccer_pipeline.py
│   │
│   ├── metrics/
│   │   ├── psnr.py
│   │   ├── ssim.py
│   │   └── lpips.py
│   │
│   └── utils/
│       ├── visualization.py
│       └── helpers.py
│
├── scripts/
│   ├── train.py                  
│   ├── eval.py                   
│   ├── benchmark.py              
│   └── export.py                 
│
├── experiments/
│   └── outputs/                  
├── pyproject.toml
└── README.md
```

## Core Libraries
- NerfStudio
- Pytorch
- W & B
- Hydra configs

## Datasets
Download the datasets from `SoccerNet-v3D`.
Then, download the actual videos from `SoccerNet-v3`.

- [SoccerNet-v3D](https://github.com/mguti97/SoccerNet-v3D)
- [SoccerNet-v3](https://github.com/SoccerNet/SoccerNet-v3)
