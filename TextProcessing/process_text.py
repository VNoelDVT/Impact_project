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
from DataMigration.database_functions import create_tables, log_to_sqlite


# Ensure NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Model Loading and Initialization
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
Qwen_25_tokenizer = AutoTokenizer.from_pretrained(model_name)
Qwen_25 = AutoModelForCausalLM.from_pretrained(model_name)

# GPU Check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Move model to GPU and set training mode
Qwen_25 = Qwen_25.to(device)
Qwen_25.train()  # Enable dropout in inference phase

# Load additional models and tokenizers
nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")

bleurt_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
bleurt_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

# Initialize ROUGE
scorer = rouge_score.rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def process_inputs(text, dropout_prob=0, num_samples=1):
    # Paramétrage du taux de dropout pour tous les modules Dropout du modèle
    for module in Qwen_25.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_prob

    # Fonction pour calculer l'entropie
    def calculate_entropy(probabilities):
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=-1)
        return entropy
    
    inputs = Qwen_25_tokenizer(text, return_tensors="pt").to(device)
        
    with torch.no_grad():
        embeddings = Qwen_25.get_input_embeddings()(inputs['input_ids'])
    
    responses = []
    token_entropies_list = []
    avg_token_entropies_list = []
    top_logits_list = []
    top_tokens_list = []
    top_probabilities_list = []
    
    for _ in range(num_samples):
        with torch.no_grad():
            output = Qwen_25.generate(**inputs, return_dict_in_generate=True, output_scores=True, do_sample=True, max_new_tokens=700)
            generated_text = Qwen_25_tokenizer.decode(output.sequences[0], skip_special_tokens=True).replace(text, '').strip()
            responses.append(generated_text)
                
            logits = torch.stack(output.scores, dim=0)  # Logits des tokens générés
            probabilities = torch.softmax(logits, dim=-1)
            
            # Calculer l'entropie des tokens
            token_entropies = calculate_entropy(probabilities)
            avg_token_entropy = token_entropies.mean().item()
            
            # Obtenir les indices et logits des top 10 pour le dernier token
            top_logits, top_indices = torch.topk(logits[-1], k=10, dim=-1)
            mask = top_logits >= 1e-3  # Filtrer les logits faibles
            filtered_logits = top_logits[mask]
            filtered_indices = top_indices[mask]
            
            filtered_probabilities = torch.softmax(filtered_logits, dim=-1) if len(filtered_logits) > 0 else []

            top_tokens = [Qwen_25_tokenizer.decode(idx) for idx in filtered_indices.tolist()]
            
            # Stocker les informations dans les listes
            token_entropies_list.append(token_entropies.cpu().numpy())
            avg_token_entropies_list.append(avg_token_entropy)
            top_logits_list.append(filtered_logits.cpu().numpy())
            top_tokens_list.append(top_tokens)
            top_probabilities_list.append(filtered_probabilities.cpu().numpy())
    
    # Assembler les données dans un DataFrame pour ce texte
    results_df = pd.DataFrame({
        'original_text': [text] * num_samples,
        'generated_response': responses,
        'embeddings': [embeddings.cpu().numpy()] * num_samples,
        'token_entropies': token_entropies_list,
        'avg_token_entropy': avg_token_entropies_list,
        'top_10_logits': top_logits_list,
        'top_10_tokens': top_tokens_list,
        'top_10_probabilities': top_probabilities_list
    })
    log_to_sqlite("process_input_table", results_df)
    
    return results_df

def calculate_all_scores(GT1, response):
    GT1 = str(GT1)
    response = str(response)

    # Pré-calcul des embeddings et tokens de référence
    inputs_gt1 = bert_tokenizer(GT1, return_tensors="pt", truncation=True)
    with torch.no_grad():
        gt1_embedding = bert_model(**inputs_gt1).last_hidden_state.mean(dim=1)
    gt1_tokens = nltk.word_tokenize(GT1)

    # --- NLI Score ---
    inputs_nli = nli_tokenizer(response, GT1, return_tensors="pt", truncation=True)
    with torch.no_grad():
        nli_logits = nli_model(**inputs_nli).logits
    nli_scores = torch.softmax(nli_logits, dim=-1).squeeze().tolist()

    # Ensure nli_scores is a list with three values
    if not isinstance(nli_scores, list):
        nli_scores = [nli_scores, 0.0, 0.0]
    if len(nli_scores) < 3:
        nli_scores += [0.0] * (3 - len(nli_scores))

    # --- BLEURT Score ---
    inputs_bleurt = bleurt_tokenizer(response, GT1, return_tensors="pt", truncation=True)
    with torch.no_grad():
        bleurt_logits = bleurt_model(**inputs_bleurt).logits
    bleurt_score = torch.softmax(bleurt_logits, dim=-1)[:, 0].item()

    # --- Cosine Similarity ---
    inputs_response = bert_tokenizer(response, return_tensors="pt", truncation=True)
    with torch.no_grad():
        response_embedding = bert_model(**inputs_response).last_hidden_state.mean(dim=1)
    cosine_score = cosine_similarity(response_embedding.cpu().numpy(), gt1_embedding.cpu().numpy())[0][0]

    # --- METEOR Score ---
    response_tokens = nltk.word_tokenize(response)
    meteor_score_result = meteor_score([gt1_tokens], response_tokens)

    # --- ROUGE-L Score ---
    rougeL_score = scorer.score(GT1, response)['rougeL'].fmeasure

    # Modify results dictionary
    results = {
        'similarity_nli_contradiction': float(nli_scores[0]),
        'similarity_nli_neutral': float(nli_scores[1]),
        'similarity_nli_entailment': float(nli_scores[2]),
        'similarity_sum_entailment_neutral': float(nli_scores[1] + nli_scores[2]),
        'similarity_bleurt': float(bleurt_score),
        'similarity_cosine': float(cosine_score),
        'meteor_score': float(meteor_score_result),
        'rougeL_score': float(rougeL_score)
    }

    log_to_sqlite("calculate_score_table", results)

    return results

def evaluate_responses(GT1, responses):
    results_list = []
    
    for i, response in enumerate(responses, start=1):
        # Calculer les scores pour la réponse actuelle
        scores = calculate_all_scores(GT1, response)
        
        # Ajouter la réponse et les scores dans la liste des résultats
        scores['generated_response'] = response
        scores['response_index'] = f"r{i}"
        results_list.append(scores)
    
    # Créer un DataFrame à partir des résultats
    final_df = pd.DataFrame(results_list)

    log_to_sqlite("evaluate_response_table", final_df)

    return final_df

# Initialize database
db_path = 'PROMPT_AND_RESULTS.db'
create_tables(db_path)