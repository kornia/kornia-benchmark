# torchdynamo/torchinductor benchmark

## Install

Create a new virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Get the latest torch version with torchdynamo and inductor

```bash
pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117
```

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python3 test_grayscale.py
```
