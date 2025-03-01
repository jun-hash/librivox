add-apt-repository -y ppa:jonathonf/ffmpeg-4
apt update
apt install -y ffmpeg
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt