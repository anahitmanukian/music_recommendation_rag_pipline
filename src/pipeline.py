from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq 
from src import config
import os



class MusicRAGPipeline:
    def __init__(self):
        print("Initializing RAG components...")
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        
        print("Loading local FAISS index...")
        self.vector_store = FAISS.load_local(
            str(config.STORAGE_DIR), 
            self.embeddings,
            allow_dangerous_deserialization=True # Required for loading local FAISS files safely
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Initialize your LLM
        self.llm = ChatGroq(
                model=config.LLM_MODEL, 
                temperature=0.3,
                groq_api_key=os.environ.get("GROQ_API_KEY")
            )
        self.chain = self._build_chain()

    def _build_chain(self):
        template = """
        You are an expert musicologist.

        Use the context to recommend songs, albums, and artists.

        When recommending a song:

        1. Explain why you chose it.
        2. Mention its musical style.
        3. Mention who might enjoy it.
        4. If the context lacks enough information, use your own music knowledge.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain

    def query(self, user_question: str) -> str:
        print(f"Searching index and generating answer for: '{user_question}'")

        docs = self.retriever.invoke(user_question)

        print("\nRetrieved Documents")
        print("=" * 50)

        for i, doc in enumerate(docs, start=1):
            print(f"\nChunk {i}")
            print(doc.page_content[:500])
            print("-" * 50)

        return self.chain.invoke(user_question)