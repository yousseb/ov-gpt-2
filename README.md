# ov-gpt-2 [![Test](https://github.com/yousseb/ov-gpt-2/actions/workflows/python.yml/badge.svg)](https://github.com/yousseb/ov-gpt-2/actions/workflows/python.yml)
OpenVino GPT-2 


## Pre-requisites
1. Python3
2. Decent hardware - for now, we rely only on CPU, but OpenVino can utilze your iGPU if present

## Installation
This should work on Windows, Linux and macOS

1. Create a venv. Follow instructions here to setup and activate the environment https://docs.python.org/3/library/venv.html

2. Clone the repository
```
git clone https://github.com/yousseb/ov-gpt-2
cd ov-gpt-2
```

3. Install requirements
```
pip3 install -r requirements.txt
```

4. Run
```
python3 main.py
```

The first time you run the application, it will 
1. Download the huggingface pytorch GPT2 model
2. Convert it to ONNX model
3. Convert the ONNX model to OpenVino IR
4. Download vocab.json and merges.txt required for GPT2
5. Start chatting


