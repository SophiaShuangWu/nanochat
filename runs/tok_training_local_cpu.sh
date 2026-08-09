export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

uv venv --allow-existing --python 3.10
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n 8 -w 16
python -m scripts.tok_train
python -m scripts.tok_eval
