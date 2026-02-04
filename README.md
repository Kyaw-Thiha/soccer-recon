# Soccer 3D Reconstruction
3D Reconstruction of soccer matches from multiview camera using `Gaussian Splatting`.

## Installation

```bash
# Clone the repository
git clone https://github.com/Kyaw-Thiha/soccer-recon.git
cd soccer-recon

# Install the package
pip install -e .

# With dev dependencies (pytest)
pip install -e ".[dev]"
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


