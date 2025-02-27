#!/bin/bash
set -e

# 1) 기본 패키지 설치 (python3, pip, ffmpeg, unzip, git, curl, tmux)
sudo apt-get update
sudo apt-get install -y python3 python3-pip ffmpeg unzip git curl tmux python3.11-venv

echo "=== 가상환경 설정 ==="
python3 -m venv venv
tmux new -s my

source venv/bin/activate
git clone https://github.com/jun-hash/librivox.git
cd librivox
pip install -r requirements.txt

python scripts/1_down_cut.py --start_idx 0 --end_idx 12269 --end_stage vad