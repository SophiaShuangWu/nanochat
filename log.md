# 2026-8-11 Update:
Reading base_train.py...

I think Karpathy just showed us the standard way to pass hyperparameters into a Python script:
```python
import argparse
parser = argparse.ArgumentParser(description="Pretrain base model")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
args = parser.parse_args()
```
To put it simply: parser.add_argument() defines the rules - it sets up what arguments are expected. Then parser.parse_args() actually enforces those rules by parsing the real command-line input. One subtle but important thing: the order matters. You can't call parse_args() before you've added all the arguments, because Python executes the script sequentially — it won't know about the rules you haven't defined yet.

I'm doing some REPL-style exploration with a tiny test.py script, running experiments on my single T600 (4GB). Here's the GPU info:
| item | value |
|-----|-----|
| CUDA available | Yes |
| CUDA version | 12.8 |
| GPU count | 1 |
| GPU 0 name | NVIDIA T600 Laptop GPU |
| GPU 0 memory | 4.29 GB |
| GPU 0 compute capability | 7.5 |
| torch version | 2.9.1+cu128 |
| torch.cuda | available |
| cuda toolkit | 12.8 |

It looks like fp16 and bfloat16 perform worse than fp32 here.



# 2026-8-10 Update:
Reading base_train.py...

I guess 
```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```
adds a key-value pair "PYTORCH_ALLOC_CONF": "expandable_segments:True" to os.environ (which is dict-like, though not a real dict).

I guess os, gc, json, time, math, argparse, dataclasses and contextlib are from the standard library, whereas wandb and torch are third-party. nanochat and scripts are our own project packages. So that's why Karpathy leaves a blank line between them — to separate imports by type.

The output of 
```python
print_banner()
```
is:

                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
which is conspicuous.


# 2026-8-9 Update:
To continue where I left off 10 days ago - I'm still working through runs/speedrun.sh. Here's the command I'm focusing on:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
```
I've been told by DeepSeek that the parameters followed by the -- delimiter are the ones passed into scripts.base_train.

I'll spend some time reading scripts/base_train.py to figure out what Karpathy originally intended, and maybe compare it with
 'Attention Is All You Need,' which I read from last May to early June.

Here's my test on 1xT600 4G GPU:
```bash
(nanochat) shuang@Shuang-PC:~/projects/llm-training-lab/nanochat$ python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=10000
```
Output on the screen:
```bash
step 09999/10000 (99.99%) | loss: 5.757840 | lrm: 0.05 | dt: 83.47ms | tok/sec: 6,134 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 14 | total time: 13.84m | eta: 0.0m
Step 10000 | Validation bpb: 1.946162
<|bos|>The capital of France is a great way to make a good idea to be a good idea to be a
<|bos|>The chemical symbol of gold is a type of material that is used in the process of the material. The process
<|bos|>If yesterday was Friday, then tomorrow will be a good idea to be a good idea to be a good idea to be a
<|bos|>The opposite of hot is the most common part of the world. The most common part of the world is
<|bos|>The planets of the solar system are: the system of the system, which is the system of the system, and the
<|bos|>My favorite color is a great way to make a good idea of a lot of people who are not
<|bos|>If 5*x + 3 = 13, then x is 1.5.5.5.5.5.5.5
2026-08-09 22:54:11,737 - nanochat.checkpoint_manager - INFO - Saved model parameters to: /home/shuang/.cache/nanochat/base_checkpoints/d4/model_010000.pt
2026-08-09 22:54:11,737 - nanochat.checkpoint_manager - INFO - Saved metadata to: /home/shuang/.cache/nanochat/base_checkpoints/d4/meta_010000.json
2026-08-09 22:54:12,023 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: /home/shuang/.cache/nanochat/base_checkpoints/d4/optim_010000_rank0.pt
Peak memory usage: 611.95MiB
Total training time: 13.84m
Minimum validation bpb: 1.944503
```
The test results are a bit hard to interpret, but they're acceptable. I feel good when I ask my T600 to train the model. I feel relaxed, like breathing fresh air while my mind is wrapped in silk. I really enjoy the implementation.


# 2026-7-30 Update:
I summarized the tokenizer training process and created tok_training_local_cpu.sh inside the /runs directory. For pretraining, I also created pretraining_local_t600.sh in the same folder.

Just a quick reminder: make sure you're in the root /nanochat directory before running these scripts:
```bash
bash runs/tok_training_local_cpu.sh
bash runs/pretraining_local_t600.sh
```

I'll sync the outputs tomorrow. (2026-7-31: Both worked locally on my PC.)


# 2026-7-29 Update:
After inspecting /scripts/tok_eval.py and /nanochat/tokenizer.py, I found the answers I was looking for.

It turns out that our newly trained tokenizer, as well as the GPT‑2 and GPT‑4 tokenizers, are all instances of tiktoken.Encoding. More specifically, they correspond to the .enc attribute of RustBPETokenizer objects in our project.

With these, we can compute the vocabulary size, encode text into token ID lists, and decode them back. This allows us to measure a useful metric: the byte-to-token ratio. Specifically, we take the UTF‑8 byte length of the input text and divide it by the number of tokens produced.

There are two ways to instantiate a tokenizer: either load a .pkl file from a given path, or call tiktoken.get_encoding() with the appropriate encoding name string.

The next question is where the evaluation texts come from. From tok_eval.py, I can clearly see there are 7 types: a news snippet, a Korean website excerpt, some of Karpathy’s own Python code, a LaTeX math snippet, a scientific paper excerpt, the first row group of the first ClimbMix shard, and the first row group of the last ClimbMix shard. The evaluation set is actually much smaller than I expected — quite tiny, honestly.

I’ve already tested the author’s code locally using test.py; feel free to check it out or replicate the same steps in your own project.

Next up is base training (or pretraining), followed by post‑training. But first, let me go check with DeepSeek.


# 2026-7-27 Update:
To proceed the reproduction pipeline, it's time to evaluate the tokenizer I trained using shards 0-7. I swtich to directory /home/shuang/projects/llm-training-lab/nanochat/, and run the following commands:
```bash
source .venv/bin/activate
python -m scripts.tok_eval
```
So I got the comparison results up on screen - it's between my tokenizer, GPT-2, and GPT-4. The main metric is compression ratio, which basically means bytes divided by tokens for each type of text. From what I can see, my tokenizer beats GPT-2 pretty clearly, but it's still not quite at GPT-4's level. What I'm curious about now is: what texts were used for evaluation, and how exactly are the GPT-2 and GPT-4 tokenizers being called under the hood? Next step is to go check /scripts/tok_eval.py.


# 2026-7-26 Update:
Here's the Python code I use to check the .parquet file, 
```python
import pandas as pd
df = pd.read_parquet('/home/shuang/.cache/nanochat/base_data_climbmix/shard_00000.parquet')
print(df.head())
print(df.info())
```
followed by the code I use for the .pkl file,
```python
import pickle
with open('/home/shuang/.cache/nanochat/tokenizer/tokenizer.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
print(data.name, data.n_vocab, data._speical_tokens, sep='\n')
tokens = data.encode("I'm Sophia")
print(tokens)
text = data.decode([73, 1653, 26383, 604])
print(text)
special_token = data.encode('|<bos>|', allowed_special={'<|bos|>'})
print(special_token)
special_text = data.decode([32759])
print(special_text)
```
then the code I use for .pt file:
```python
import os
import torch
data = torch.load('/home/shuang/.cache/nanochat/tokenizer/token_bytes.pt')
print(type(data))
print(data.shape)
filename = 'token_bytes.txt'
file_path = os.path.abspath(filename)
with open(file_path, 'w') as f:
    for value in data:
        f.write(f'{value}\n')
print(f'{filename} has been saved to {file_path}')
```
Actually, let's circle back to /scripts/tok_train.py and see where those two files came from.
- tokenizer.pkl:
```python
tokenizer.save(tokenizer_dir)
```
- token_bytes.pt:
```python
torch.save(token_bytes, f)
```
So the tokenizer, which is an Encoding object named "rustbpe", has been saved into tokenizer.pkl. It contains a vocabulary with 2**15 tokens. We can call the encode function to encode any string into a list of token_ids corresponding to tokens in the vocabulary. A token is actually a limited number of bytes that are tightly packed from left to right in a solid way. We can also call the decode function to decode a list of token_ids back to text. The route should be: token_id list -> byte sequence -> utf-8 decoding -> text. The token_bytes.pt file keeps a torch.tensor object, which is a vector recording the length of each token corresponding to the token_id in the vocabulary.

I can dive into the rustbpe module and the tiktoken module required in /nanochat/tokenizer.py with full interest and enthusiasm, but I need to return to the reproduction pipeline. I'll leave that part for later when I'm about to rewrite the whole project. Until now, dataset download and tokenizer training have become quite clear to me. I'll proceed with /runs/speedrun.sh tomorrow.


# 2026-7-25 Update:
Current diretory is ~/projects/llm-training-lab/nanochat. I use the following commands to do the tokenizer training:
```bash
source .venv/bin/activate
python -m scripts/tok_train
```

Alright so I made this cache/ folder with two joints in it:
- base_data_climbix/ — holds the .parquet files from downloading the dataset
- tokenizer/ — holds the .pkl file and .pt file from training the tokenizer

I'm about to kick off the base training, but before that I just wanna take a look at what's actually in these files, you know? Make sure nothing's cooked.

So I threw together some Python scripts to open 'em up and peek inside. I'll drop those scripts in here tomorrow. For now, here's my understanding of what the tokenizer even does and how it does it.

- A token is a sequence of bytes.
- A token ID is the integer index of that token in the vocabulary.
- Most tokens are not single bytes. They are formed by repeatedly merging adjacent byte pairs during BPE training.
- A byte is a bytes object in Python, with values from 0 to 255. It is written as b'\xHH', where HH is a two‑digit hexadecimal number.
- Any string (including emojis, Chinese characters, etc.) can be encoded into a sequence of bytes using UTF‑8:
    - 'a' → b'a'
    - '😊' → b'\xf0\x9f\x98\x8a' (multiple bytes)
- The tokenizer takes a raw text string, encodes it to UTF‑8 bytes, and then walks through that byte sequence from left to right.
- At each position, it uses a Trie built from the vocabulary to find the longest token that matches the current byte sequence.
- It consumes that token, then continues from the next byte.
- The output is an ordered list of tokens (or token IDs), which is the final encoded representation of the original text.

# 2026-7-24 Update:
I'm working with DeepSeek and ChatGPT on runs/speedrun.sh. To keep the connection to ChatGPT stable, I have to switch Clash Verge from rule-based mode to global mode.

I've decided to implement the tokenizer locally. First, I just did the prep following runs/speedrun.sh.
```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
uv sync
```

I'm not running WandB locally, but I already got the API key(https://wandb.ai/profile/sophiashuangwu, https://wandb.ai/settings#apikeys) set up for when I do the training on the cloud. 
```bash
wandb login
```

I checked out nanochat/dataset.py and realized it's going to download shards 0–7 for training and the last shard for validation. So I cd'd into ~/projects/llm-training-lab/nanochat and ran these commands:
```bash
source .venv/bin/activate
python -m nanochat.dataset -n 8 -w 16
```


# 2026-7-23 Update:
The first reference is not that clear, and it still recommands Lamda Cloud. So I'm switching to the second reference.

I'm using google-colab-cli now. Successfully fixed the issue where WSL couldn't connect through Clash Verge.

Troubleshooting references:
https://jiahhhao.github.io/2025/11/20/WSL2%E9%80%9A%E8%BF%87Clash-Verge%E8%BF%9E%E6%8E%A5%E5%A4%96%E7%BD%91/
https://clashofficial.com/zh-CN/blog/articles/clash-wsl2-windows-proxy-mirrored-networking.html

The key is to enable LAN setting and find the mix port, like 7897, in clash verge. Then use the command 
```bash
ipconfig
``` 
to find the ipv4 corresponding to WSL. Switch to WSL and test the command
```bash 
timeout 2 bash -c "echo > /dev/tcp/XXX.XX.XXX.X/7897" && echo "✅" || echo "❌"
```
it should return ✅. Then paste the following code in ~/.bashrc:
```bash
# Obtain a Windows IP address and set up a proxy (including case sensitivity settings).
export WIN_IP=$(ip route show default | awk '{print $3}')
export HTTP_PROXY="http://$WIN_IP:7897"
export HTTPS_PROXY="http://$WIN_IP:7897"
export http_proxy="http://$WIN_IP:7897"
export https_proxy="http://$WIN_IP:7897"
```

Return to WSL and use the command 
```bash
source ~/.bashrc
```
Then test the command 
```bash
curl -I https://www.google.com
```
It should return HTTP/2 200.

Actually I used the command 
```bash
uv tool install google-colab-cli 
```
for installation, and the command 
```bash
colab new -s Sophia --gpu T4
```
for session start.

The current issue is that runs/speedrun.sh is designed for a single node with 8×H100 GPUs, but I only have access to a free 1×T4 GPU. So I need to understand the script first and then adapt it to my limited hardware.


# 2026-7-22 Update:
I want to reproduce the training of nanochat on Colab. 

I've tried using Lambda Cloud but I don't have a credit card so I'm switching to free Colab.

I need some more information about whether this is feasible. So I Bing'd the question and found the following references. 

Refernces:
https://yomxxx.com/posts/2026-05-29-nanochat-karpathy-full-stack-llm-training
https://blog.csdn.net/weixin_44151034/article/details/159589896

It's 22:43PM, I'll continue the work tomorrow.