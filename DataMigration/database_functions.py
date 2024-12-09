import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
from nltk.translate.meteor_score import meteor_score
import nltk
from rouge_score import rouge_scorer
import rouge_score
import sqlite3
from contextlib import contextmanager
from openai import OpenAI



# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

@contextmanager
def sqlite_connection(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def create_tables(db_path):
    with sqlite_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS process_input_table (
                original_text TEXT,
                generated_response TEXT,
                avg_token_entropy REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calculate_score_table (
                similarity_nli_contradiction REAL,
                similarity_nli_neutral REAL,
                similarity_nli_entailment REAL,
                similarity_sum_entailment_neutral REAL,
                similarity_bleurt REAL,
                similarity_cosine REAL,
                meteor_score REAL,
                rougeL_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluate_response_table (
                generated_response TEXT,
                response_index TEXT,
                similarity_nli_contradiction REAL,
                similarity_nli_neutral REAL,
                similarity_nli_entailment REAL,
                similarity_sum_entailment_neutral REAL,
                similarity_bleurt REAL,
                similarity_cosine REAL,
                meteor_score REAL,
                rougeL_score REAL
            )
        ''')
        
        conn.commit()

def log_to_sqlite(table_name, data):
    db_path = 'PROMPT_AND_RESULTS.db'
    with sqlite_connection(db_path) as conn:
        cursor = conn.cursor()
        if table_name == "process_input_table":
            for index, row in data.iterrows():
                cursor.execute('''
                    INSERT INTO process_input_table (
                        original_text, generated_response,
                        avg_token_entropy
                    ) VALUES (?, ?, ?)
                ''', (
                    row['original_text'], 
                    row['generated_response'], 
                    #row['embeddings'].tobytes() if hasattr(row['embeddings'], 'tobytes') else None,
                    #row['token_entropies'].tobytes() if hasattr(row['token_entropies'], 'tobytes') else None, 
                    row['avg_token_entropy'],
                    #row['top_10_logits'].tobytes() if hasattr(row['top_10_logits'], 'tobytes') else None, 
                    #','.join(row['top_10_tokens']),
                    #row['top_10_probabilities'].tobytes() if hasattr(row['top_10_probabilities'], 'tobytes') else None
                ))
        elif table_name == "calculate_score_table":
            cursor.execute('''
                INSERT INTO calculate_score_table (
                    similarity_nli_contradiction, similarity_nli_neutral, similarity_nli_entailment,
                    similarity_sum_entailment_neutral, similarity_bleurt, similarity_cosine,
                    meteor_score, rougeL_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get('similarity_nli_contradiction', 0),
                data.get('similarity_nli_neutral', 0),
                data.get('similarity_nli_entailment', 0),
                data.get('similarity_sum_entailment_neutral', 0),
                data.get('similarity_bleurt', 0),
                data.get('similarity_cosine', 0),
                data.get('meteor_score', 0),
                data.get('rougeL_score', 0)
            ))
        elif table_name == "evaluate_response_table":
            for index, row in data.iterrows():
                cursor.execute('''
                    INSERT INTO evaluate_response_table (
                        generated_response, response_index, 
                        similarity_nli_contradiction, similarity_nli_neutral, 
                        similarity_nli_entailment, similarity_sum_entailment_neutral, 
                        similarity_bleurt, similarity_cosine, 
                        meteor_score, rougeL_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['generated_response'], 
                    row['response_index'],
                    row.get('similarity_nli_contradiction', 0),
                    row.get('similarity_nli_neutral', 0),
                    row.get('similarity_nli_entailment', 0),
                    row.get('similarity_sum_entailment_neutral', 0),
                    row.get('similarity_bleurt', 0),
                    row.get('similarity_cosine', 0),
                    row.get('meteor_score', 0),
                    row.get('rougeL_score', 0)
                ))
        conn.commit()