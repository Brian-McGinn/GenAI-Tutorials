# Retrieval Augmented Generation (RAG) Tutorial

AI are trained on large sets of information, but that information can quickly become outdated or may not cover the specific details you care about. For example, if you ask an AI about a recent event or something unique to your company, it might not know the answer or could even make up a response that sounds correct but isn’t accurate. This is known as a “hallucination,” and it’s a common challenge with AIs.

RAG helps solve this problem by adding an additoinal step that uses an AI to search for relavant data in real time and use that information to answer questions. Instead of guessing, the AI can pull up the exact details from your files, policies, or records, making its answers more reliable and easier to verify.

To enable RAG, an important step is preparing your data so it is searchable. This usually means storing your documents in a format that allows the AI to quickly retrieve specific passages. In practice, this could involve setting up a specialized database for documents (often a “vector database”) or indexing your existing content in a way that supports efficient lookups. In some cases, it may also involve cleaning or restructuring your data to ensure consistency.


Use the collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brian-McGinn/GenAI-Tutorials/blob/main/rag-tutorial/RAG_Tutorial.ipynb)

or Run locally by following the below steps.

## Python virtual envrionment setup

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Jupyter Notebooks for local execution

```bash
pip install jupyter
```

## Generate a Together AI API Key

Follow Together AI [quickstart](https://docs.together.ai/docs/quickstart)


## Run notebook

```bash
jupyter notebook
```

Follow steps in [Notebook](http://localhost:8888/notebooks/rag-tutorial/RAG_Tutorial.ipynb)
