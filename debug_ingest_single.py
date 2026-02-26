from langchain_community.document_loaders import TextLoader
import traceback
import os

file_path = "data/lorundrostat.txt"
print(f"Testing loading {file_path}...")

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        print(f"Successfully loaded {len(docs)} documents.")
        print(docs[0].page_content[:100])
    except Exception:
        traceback.print_exc()
