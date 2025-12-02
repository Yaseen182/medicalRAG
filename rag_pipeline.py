"""
RAG Pipeline implementation using LangChain with local Qwen3 4B model
Handles document chunking, embedding, vector storage, and retrieval
"""
import torch
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MedicalRAGPipeline:
    """RAG Pipeline for Medical Chatbot using local LLM"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_path: str = "./data/vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: str = "cuda",
        top_k: int = 5
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.vector_store_path = Path(vector_store_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device if torch.cuda.is_available() else "cpu"
        self.top_k = top_k
        
        logger.info(f"Initializing RAG Pipeline with device: {self.device}")
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        model_kwargs = {'device': self.device}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        logger.info("Embedding model loaded successfully")
    
    def _initialize_llm(self):
        """Initialize local Qwen3 4B model"""
        if self.llm is not None:
            logger.info("LLM already initialized")
            return
        
        logger.info(f"Loading LLM: {self.model_name}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                do_sample=True,
                return_full_text=False
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunk documents into smaller pieces for embedding
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Chunking {len(documents)} documents...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            metadata["source"] = doc.get("source", "unknown")
            
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                
                chunked_docs.append(
                    Document(page_content=chunk, metadata=chunk_metadata)
                )
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs
    
    def create_vector_store(self, documents: List[Document], store_type: str = "faiss"):
        """
        Create vector store from documents
        
        Args:
            documents: List of LangChain Document objects
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        logger.info(f"Creating {store_type} vector store with {len(documents)} documents...")
        
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if store_type.lower() == "faiss":
                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                # Save FAISS index
                self.vectorstore.save_local(str(self.vector_store_path / "faiss_index"))
                
            elif store_type.lower() == "chroma":
                self.vectorstore = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=str(self.vector_store_path / "chroma_db")
                )
                self.vectorstore.persist()
            
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            logger.info(f"Vector store created and saved to {self.vector_store_path}")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self, store_type: str = "faiss"):
        """
        Load existing vector store
        
        Args:
            store_type: Type of vector store ('faiss' or 'chroma')
        """
        logger.info(f"Loading {store_type} vector store from {self.vector_store_path}...")
        
        try:
            if store_type.lower() == "faiss":
                index_path = self.vector_store_path / "faiss_index"
                if not index_path.exists():
                    raise FileNotFoundError(f"FAISS index not found at {index_path}")
                
                self.vectorstore = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
            elif store_type.lower() == "chroma":
                chroma_path = self.vector_store_path / "chroma_db"
                if not chroma_path.exists():
                    raise FileNotFoundError(f"Chroma DB not found at {chroma_path}")
                
                self.vectorstore = Chroma(
                    persist_directory=str(chroma_path),
                    embedding_function=self.embeddings
                )
                
                # Get Chroma collection stats
                try:
                    collection = self.vectorstore._collection
                    count = collection.count()
                    logger.info(f"Vector store loaded successfully with {count} documents")
                except Exception as e:
                    logger.warning(f"Could not get Chroma collection count: {e}")
                    logger.info("Vector store loaded successfully")
            
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")
            
            # Log vector store stats for FAISS
            if store_type.lower() == "faiss" and hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                logger.info(f"Vector store loaded successfully with {self.vectorstore.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        logger.info("Setting up QA chain...")
        
        # Initialize LLM if not already done
        self._initialize_llm()
        
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Load or create vector store first.")
        
        # Custom prompt template optimized for Qwen model
        prompt_template = """<|im_start|>system
You are MediChat, a helpful medical assistant AI. Provide clear, accurate medical information based on the context given. Keep answers concise and professional.
If the question is not about medical field say "I can't help with this type of questions, I'm medical chatbot".
Also, you can reply welcoming words like "Hey, hello, good by, thanks" <|im_end|>
<|im_start|>user
Context information:
{context}

Question: {question}<|im_end|>
<|im_start|>assistant
"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA chain setup complete")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        logger.info(f"Processing query: {question[:100]}...")
        
        try:
            # First, manually retrieve source documents
            if self.vectorstore is None:
                raise ValueError("Vector store not initialized")
            
            # Check vector store stats
            if hasattr(self.vectorstore, 'index') and hasattr(self.vectorstore.index, 'ntotal'):
                logger.info(f"Vector store contains {self.vectorstore.index.ntotal} vectors")
            
            # Retrieve documents using similarity search
            logger.info(f"Searching for top {self.top_k} similar documents...")
            retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents from vector store")
            
            if retrieved_docs:
                logger.debug(f"First retrieved doc metadata: {retrieved_docs[0].metadata}")
                logger.debug(f"First retrieved doc content preview: {retrieved_docs[0].page_content[:100]}...")
            else:
                logger.error("No documents retrieved! Vector store may be empty or query failed.")
            
            # Now run the QA chain
            result = self.qa_chain({"query": question})
            
            # Extract source documents
            source_docs = result.get("source_documents", [])
            
            # Use manually retrieved docs if qa_chain didn't return them
            if not source_docs and retrieved_docs:
                logger.warning("QA chain returned no source documents, using manually retrieved docs")
                source_docs = retrieved_docs
            
            logger.info(f"Using {len(source_docs)} source documents")
            
            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in source_docs
                ]
            }
            
            logger.info(f"Query processed successfully with {len(response['source_documents'])} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search without LLM generation
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        logger.info(f"Performing similarity search for: {query[:100]}...")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return results


def initialize_rag_pipeline(config) -> MedicalRAGPipeline:
    """
    Initialize RAG pipeline with configuration
    
    Args:
        config: Application configuration object
        
    Returns:
        Initialized MedicalRAGPipeline instance
    """
    pipeline = MedicalRAGPipeline(
        model_name=config.LLM_MODEL_NAME,
        embedding_model=config.EMBEDDING_MODEL,
        vector_store_path=config.VECTOR_STORE_PATH,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        device=config.LLM_DEVICE,
        top_k=config.TOP_K_RESULTS
    )
    
    return pipeline


if __name__ == "__main__":
    # Test RAG pipeline
    from logger import setup_logging
    from config import get_settings
    from data_loader import MedicalDataLoader, prepare_documents_for_rag
    
    setup_logging()
    config = get_settings()
    
    # Load data
    loader = MedicalDataLoader()
    documents = loader.load_processed_data()
    prepared_docs = prepare_documents_for_rag(documents[:100])  # Test with first 100
    
    # Initialize pipeline
    rag = initialize_rag_pipeline(config)
    
    # Create vector store
    chunked_docs = rag.chunk_documents(prepared_docs)
    rag.create_vector_store(chunked_docs)
    
    # Test query
    response = rag.query("What are the symptoms of diabetes?")
    print(response)
