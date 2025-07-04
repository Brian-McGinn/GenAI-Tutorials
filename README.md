# fine-tuning
jupyter notebook

## Python virtual envrionment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install jupyter
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-r1:1.5b &
ollama serve
ipython kernel install --user --name=venv
jupyter notebook
```

## Run notebook

Follow steps in [Notebook](http://localhost:8888/)
