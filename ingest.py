
import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print("Loading documents...")
    documents = []

    # 1. Load PDF files
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            print(f"Loaded PDF: {pdf_file}")
        except Exception as e:
            print(f"Error loading PDF {pdf_file}: {e}")

    # 2. Load TXT files (Explicitly using UTF-8)
    txt_files = glob.glob(os.path.join(DATA_PATH, "*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded TXT: {txt_file}")
        except Exception as e:
             print(f"Error loading TXT {txt_file}: {e}")
             
    # 3. Load MD files (Explicitly using UTF-8)
    md_files = glob.glob(os.path.join(DATA_PATH, "*.md"))
    for md_file in md_files:
        try:
            loader = TextLoader(md_file, encoding='utf-8')
            documents.extend(loader.load())
            print(f"Loaded MD: {md_file}")
        except Exception as e:
             print(f"Error loading MD {md_file}: {e}")

    if not documents:
        print("No documents found!")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks.")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    print("Building vector store...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
