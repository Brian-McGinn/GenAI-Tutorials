# LLM Fine Tuning

LLMs have become a powerful tool for applications but can have issues when moving into specific domains. To improve accuracy and performance it is common to fine tune a foundation model with your own dataset. The following jupyter notebook will go through the process of fine tuning a model using custom data. This includes generating training data and utilizing QLoRA for quick and flexible training.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brian-McGinn/Fine-Tuning-Tutorial/blob/main/Fine-Tuning-Tutorial.ipynb)


## Installing CUDA

If running locally you will need to have pytorch and cuda installed. You can follow the [PyTorch setup steps](https://pytorch.org/get-started/locally/)

## Python virtual envrionment setup

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Jupyter Notebooks for local execution

```bash
pip install jupyter
```

## Install Ollama for local training

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.1 &
ipython kernel install --user --name=venv
```

## Run notebook

```bash
jupyter notebook
```

Follow steps in [Notebook](http://localhost:8888/doc/tree/Fine-Tuning-Tutorial.ipynb)
