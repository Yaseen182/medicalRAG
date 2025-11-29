"""
Data ingestion pipeline for medical datasets
Handles loading and preprocessing of MedQA, MedDialog, HealthSearchQA, and LiveQA Medical datasets
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """Loads and preprocesses medical datasets"""
    
    def __init__(self, data_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_medqa(self) -> List[Dict[str, Any]]:
        """
        Load MedQA dataset from HuggingFace
        Medical question answering dataset based on medical licensing exams
        """
        logger.info("Loading MedQA dataset...")
        try:
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="train", trust_remote_code=True)
            
            documents = []
            for item in tqdm(dataset, desc="Processing MedQA"):
                doc = {
                    "source": "MedQA",
                    "type": "qa",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "options": item.get("options", {}),
                    "metadata": {
                        "dataset": "MedQA",
                        "exam_type": "medical_licensing"
                    }
                }
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from MedQA")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading MedQA: {e}")
            return []
    
    def load_meddialog(self) -> List[Dict[str, Any]]:
        """
        Load MedDialog dataset
        Doctor-patient conversation dataset
        """
        logger.info("Loading MedDialog dataset...")
        try:
            dataset = load_dataset("medical_dialog", "processed.en", split="train", trust_remote_code=True)
            
            documents = []
            for item in tqdm(dataset, desc="Processing MedDialog"):
                # MedDialog contains conversations
                utterances = item.get("utterances", [])
                if utterances:
                    doc = {
                        "source": "MedDialog",
                        "type": "dialogue",
                        "conversation": utterances,
                        "description": item.get("description", ""),
                        "metadata": {
                            "dataset": "MedDialog",
                            "conversation_type": "doctor_patient"
                        }
                    }
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from MedDialog")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading MedDialog: {e}")
            return []
    
    def load_healthsearchqa(self) -> List[Dict[str, Any]]:
        """
        Load HealthSearchQA dataset (BI55/MedText)
        Medical question-answer dataset with Prompt and Completion columns
        """
        logger.info("Loading HealthSearchQA dataset...")
        try:
            # Load BI55/MedText dataset which has Prompt and Completion columns
            dataset = load_dataset("BI55/MedText", split="train")
            
            documents = []
            for item in tqdm(dataset, desc="Processing HealthSearchQA"):
                # Extract Prompt and Completion from the dataset
                question = item.get("Prompt", "")
                answer = item.get("Completion", "")
                
                if question and answer:
                    doc = {
                        "source": "HealthSearchQA",
                        "type": "qa",
                        "question": question,
                        "answer": answer,
                        "metadata": {
                            "dataset": "HealthSearchQA",
                            "query_type": "health_search"
                        }
                    }
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from HealthSearchQA")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading HealthSearchQA: {e}")
            return []
    
    def load_liveqa_medical(self) -> List[Dict[str, Any]]:
        """
        Load LiveQA Medical dataset
        Consumer health questions with expert answers
        """
        logger.info("Loading LiveQA Medical dataset...")
        try:
            # Use the correct dataset path with test split
            dataset = load_dataset("hyesunyun/liveqa_medical_trec2017", split="test")
            
            documents = []
            for item in tqdm(dataset, desc="Processing LiveQA"):
                question = item.get("question", item.get("QUESTION", ""))
                answer = item.get("answer", item.get("ANSWER", ""))
                
                if question and answer:
                    doc = {
                        "source": "LiveQA",
                        "type": "qa",
                        "question": question,
                        "answer": answer,
                        "metadata": {
                            "dataset": "LiveQA",
                            "query_type": "consumer_health"
                        }
                    }
                    documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from LiveQA")
            return documents
        
        except Exception as e:
            logger.error(f"Error loading LiveQA Medical: {e}")
            logger.warning("Using fallback structure for LiveQA")
            return []
    
    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """Load all medical datasets and combine them"""
        logger.info("Loading all medical datasets...")
        
        all_documents = []
        
        # Load each dataset
        all_documents.extend(self.load_medqa())
        all_documents.extend(self.load_meddialog())
        all_documents.extend(self.load_healthsearchqa())
        all_documents.extend(self.load_liveqa_medical())
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        
        # Save combined dataset
        self.save_processed_data(all_documents)
        
        return all_documents
    
    def save_processed_data(self, documents: List[Dict[str, Any]]):
        """Save processed documents to disk"""
        output_file = self.processed_dir / "combined_medical_data.json"
        
        logger.info(f"Saving {len(documents)} documents to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info("Data saved successfully")
    
    def load_processed_data(self) -> List[Dict[str, Any]]:
        """Load previously processed data"""
        input_file = self.processed_dir / "combined_medical_data.json"
        
        if not input_file.exists():
            logger.warning("No processed data found. Loading datasets from source...")
            return self.load_all_datasets()
        
        logger.info(f"Loading processed data from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents


def prepare_documents_for_rag(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Prepare documents for RAG by formatting them appropriately
    
    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    logger.info("Preparing documents for RAG...")
    
    prepared_docs = []
    
    for doc in tqdm(documents, desc="Formatting documents"):
        try:
            if doc["type"] == "qa":
                # Format Q&A documents
                text = f"Question: {doc['question']}\n\nAnswer: {doc['answer']}"
                
            elif doc["type"] == "dialogue":
                # Format dialogue documents
                conversation = doc.get("conversation", [])
                text = doc.get("description", "") + "\n\n"
                
                # Handle different conversation formats
                if conversation:
                    if isinstance(conversation[0], dict):
                        # Dictionary format
                        text += "\n".join([f"{turn.get('speaker', 'Unknown')}: {turn.get('utterance', '')}" 
                                          for turn in conversation])
                    elif isinstance(conversation[0], str):
                        # String format
                        text += "\n".join(conversation)
                    else:
                        # Unknown format, convert to string
                        text += str(conversation)
            else:
                # Generic format
                text = str(doc)
            
            prepared_docs.append({
                "text": text,
                "metadata": doc.get("metadata", {}),
                "source": doc.get("source", "unknown")
            })
        except Exception as e:
            logger.warning(f"Error processing document: {e}")
            continue
    
    logger.info(f"Prepared {len(prepared_docs)} documents")
    return prepared_docs


if __name__ == "__main__":
    # Test data loading
    from logger import setup_logging
    setup_logging()
    
    loader = MedicalDataLoader()
    documents = loader.load_all_datasets()
    prepared = prepare_documents_for_rag(documents)
    
    print(f"Loaded and prepared {len(prepared)} documents")
