# Remote Training on Vast.ai

## 1. Installation

SSH into remote:
```bash
ssh root@<REMOTE_IP>
git clone https://github.com/Kyaw-Thiha/soccer-recon.git
cd soccer-recon
conda create -n soccer-recon python=3.12 -y
conda activate soccer-recon
conda install -c nvidia cuda-nvcc cuda-toolkit -y
pip install -e .
wandb login
```

## 2. Upload Dataset

Find a match locally:
```bash
find ~/Documents/Projects/soccer-recon/data/SoccerNet-v3D -type d -name "*-*" | head
```

Upload match data (labels + frames only):
```bash
MATCH="england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea"
REMOTE="root@<REMOTE_IP>"

ssh $REMOTE "mkdir -p /root/soccer-recon/data/SoccerNet-v3D/$(basename '$MATCH')"
scp ~/Documents/Projects/soccer-recon/data/SoccerNet-v3D/$MATCH/Labels-v3D.json $REMOTE:/root/soccer-recon/data/SoccerNet-v3D/$(basename "$MATCH")/
scp ~/Documents/Projects/soccer-recon/data/SoccerNet-v3D/$MATCH/v3/Frames/*.png $REMOTE:/root/soccer-recon/data/SoccerNet-v3D/$(basename "$MATCH")/
```

## 3. Training

SSH and run training:
```bash
ssh root@<REMOTE_IP>
cd /root/soccer-recon
conda activate soccer-recon

# Use tmux to keep it running if SSH drops
tmux new-session -d -s training
tmux send-keys -t training "python scripts/train.py --data data/SoccerNet-v3D --match-path 'england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea' --action-id 0 --experiment-name exp1 --vis wandb" Enter
```

Monitor on [wandb.ai](https://wandb.ai) in real-time.

## 4. Download Results

After training completes:
```bash
scp -r root@<REMOTE_IP>:/root/soccer-recon/outputs/exp1 ~/Documents/Projects/soccer-recon/outputs/
```

Done!
