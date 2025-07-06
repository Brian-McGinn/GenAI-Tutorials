# fine-tuning
jupyter notebook

## Python virtual envrionment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install jupyter
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-r1:1.5b &
ollama pull llama3.1
ollama serve
ipython kernel install --user --name=venv
export HUGGING_FACE_READ_KEY=<your_key>
jupyter notebook
```

## Run notebook

Follow steps in [Notebook](http://localhost:8888/)

## Installing CUDA

pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install filelock==3.16.1
pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install --force-reinstall numpy==1.26.2
pip install flatbuffers