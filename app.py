from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import AutoTokenizer as LLaMATokenizer
from sqlalchemy import create_engine, text
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset

# Load the dataset with trust_remote_code=True
dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)

# Step 2: Initialize RAG Components
tokenizer = LLaMATokenizer.from_pretrained("facebook/llama-3.1")
retriever = RagRetriever.from_pretrained("facebook/llama-3.1", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/llama-3.1", retriever=retriever)