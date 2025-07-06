import ollama
from PyPDF2 import PdfReader
import os

# Load and extract text from the PDF
def load_pdf_content(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text_content = ""
    for page in reader.pages:
        text_content += page.extract_text() + "\n"
    return text_content

# Load the unsloth documentation
pdf_content = load_pdf_content("unsloth_documentation.pdf")

# Create the RAG-enhanced prompt
rag_prompt = f"""Based on the following documentation about Unsloth.ai, please answer the user's question:

Documentation:
{pdf_content}

User Question: Give me some Unsloth.ai News?

Please provide a comprehensive answer based on the documentation provid d."""

response = ollama.chat(
    model="llama3.1",
    messages=[
        {"role": "user", "content": rag_prompt},
    ],
)
print(response["message"]["content"])