from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src import config

class MusicDataIngestor:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        
    def run_ingestion(self):
        print(f"Scanning data folder for text and PDF reviews in {config.DATA_DIR}...")
        
        documents = []
        
        # 1. Load all text files seamlessly
        txt_loader = DirectoryLoader(
            config.DATA_DIR, 
            glob="**/*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        
        documents.extend(txt_loader.load())
        
        # 2. Load all PDF files seamlessly
        pdf_loader = DirectoryLoader(
            config.DATA_DIR, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        if not documents:
            print("No documents were loaded. Check your data directory paths!")
            return

        print(f"\nSuccessfully loaded {len(documents)} total data components.")
        
        print("Splitting documents into cohesive paragraphs/chunks...")
        chunks = self.text_splitter.split_documents(documents)
        
        print("Generating embeddings and updating your local FAISS index...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        
        print(f"Saving updated FAISS index to {config.STORAGE_DIR}...")
        vector_store.save_local(str(config.STORAGE_DIR))
        print("Ingestion pipeline updated and complete!")  