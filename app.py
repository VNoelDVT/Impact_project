import streamlit as st
import pandas as pd
from datetime import datetime
import sqlite3
from TextProcessing.process_text import process_inputs, evaluate_responses
from DataMigration.database_functions import create_tables, log_to_sqlite
import io

st.set_page_config(layout="wide")  # Utiliser la mise en page large pour optimiser l'espace

st.title("Impact_V0_Chatbot")



# Initialisation des états de session
if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

if "analysis_inputs" not in st.session_state:
    st.session_state.analysis_inputs = {}

# Initialisation de la base de données SQLite
conn = sqlite3.connect('chatbot_data.db', check_same_thread=False)
c = conn.cursor()

# Création de la table si elle n'existe pas
c.execute('''CREATE TABLE IF NOT EXISTS analysis_data
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              conversation_id INTEGER,
              generated_response TEXT,
              respect_context TEXT,
              completion TEXT,
              good_practices TEXT,
              no_aberration TEXT,
              comment TEXT)''')
conn.commit()

# Créer une mise en page avec deux colonnes : gauche pour l'historique et conversation, droite pour l'analyse
col1, col2 = st.columns([3, 1])  # Ratio ajustable

# Colonne de gauche : Historique et Conversation
with col1:
    # Sidebar pour l'historique des conversations
    st.sidebar.title("Historique des Conversations")

    # Bouton pour créer une nouvelle conversation
    if st.sidebar.button("Nouvelle Conversation"):
        new_conversation = {
            "id": len(st.session_state.conversations) + 1,
            "messages": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.conversations.append(new_conversation)
        st.session_state.current_conversation = new_conversation

    # Liste des conversations précédentes
    for conv in st.session_state.conversations:
        conv_title = f"Conversation {conv['id']} - {conv['timestamp']}"
        if st.sidebar.button(conv_title):
            st.session_state.current_conversation = conv

    # Vérifier si une conversation est sélectionnée
    if st.session_state.current_conversation is None:
        st.error("Veuillez créer ou sélectionner une conversation.")
    else:
        # Affichage des messages de la conversation actuelle
        for message in st.session_state.current_conversation["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Gestion de l'entrée utilisateur
        if prompt := st.chat_input("What is up?"):
            st.session_state.current_conversation["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Génération de la réponse avec vos fonctions
            with st.chat_message("assistant"):
                # Utilisation de process_inputs pour générer des réponses
                result_df = process_inputs(prompt)
                responses = result_df["generated_response"].tolist()
                
                # Charger le ground truth
                GT = pd.read_excel("PROMPT AND GROUND TRUTH.xlsx")
                outputs_GT = GT["Output"].tolist()
                
                # Liste pour stocker toutes les évaluations
                all_evaluations = []
                
                # Boucle d'évaluation pour chaque output du GT
                for idx, gt_output in enumerate(outputs_GT, start=1):
                    eval_df = evaluate_responses(gt_output, responses)
                    eval_df = eval_df.rename(
                        columns=lambda col: f"{col}_GT{idx}" if col not in ['generated_response'] else col
                    )
                    all_evaluations.append(eval_df)
                
                # Fusionner toutes les évaluations
                final_eval_df = all_evaluations[0]
                for df in all_evaluations[1:]:
                    final_eval_df = final_eval_df.merge(df, on='generated_response', how='inner')
                
                # Sélectionner uniquement les colonnes NLI et ROUGE-L
                selected_columns = [col for col in final_eval_df.columns if 'nli' in col.lower() or 'rougeL_score' in col.lower()]
                selected_columns.insert(0, 'generated_response')
                display_df = final_eval_df[selected_columns]
                
                # Ajouter les réponses générées comme messages du bot
                st.session_state.current_response = responses  # Stocker toutes les réponses
                for response in responses:
                    st.session_state.current_conversation["messages"].append({"role": "assistant", "content": response})
                    with st.container():
                        st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
                
                # Affichage du tableau d'évaluation
                st.subheader("Résumé des évaluations (NLI et ROUGE-L) :")
                st.dataframe(display_df)

# Colonne de droite : Analyse permanente
with col2:

    st.markdown("[Aller vers le guide de règles d'analyse](https://docs.google.com/document/d/1bOZQpMz9WCxU9sDIK387HmA2iInsrehfl7Xl5deuSYY/edit?tab=t.0#heading=h.jht8l0lr586l)")
    st.header("Analyse de la Réponse")
    
    if hasattr(st.session_state, 'current_response') and st.session_state.current_response:
        # Formulaire d'analyse
        with st.form(key='analysis_form',clear_on_submit=True ):
            analysis_data = []
            for idx, response in enumerate(st.session_state.current_response, start=1):
                st.write(f"**Analyse de la Réponse {idx} :**")
                st.text_area(f"Output (Réponse {idx})", response, height=100, disabled=True)

                # Utiliser des valeurs par défaut stockées dans st.session_state
                respect_context = st.session_state.analysis_inputs.get(f"context_{idx}", "")
                completion = st.session_state.analysis_inputs.get(f"completion_{idx}", "")
                good_practices = st.session_state.analysis_inputs.get(f"practices_{idx}", "")
                no_aberration = st.session_state.analysis_inputs.get(f"aberration_{idx}", "")
                comment = st.session_state.analysis_inputs.get(f"comment_{idx}", "")

                # Champs d'analyse
                respect_context = st.text_input(f"Respect du contexte (de 0 à 25)", value=respect_context, key=f"context_{idx}")
                completion = st.text_input(f"Complétion de la réponse (de 0 à 25)", value=completion, key=f"completion_{idx}")
                good_practices = st.text_input(f"Respect des bonnes pratiques (de 0 à 25)", value=good_practices, key=f"practices_{idx}")
                no_aberration = st.text_input(f"Absence d’aberration (de 0 à 25)", value=no_aberration, key=f"aberration_{idx}")
                comment = st.text_area(f"Commentaire (Texte)", value=comment, key=f"comment_{idx}")

                # Collecter les données d'analyse
                analysis_data.append({
                    'generated_response': response,
                    'Respect du contexte': respect_context,
                    'Complétion de la réponse': completion,
                    'Respect des bonnes pratiques': good_practices,
                    'Absence d’aberration': no_aberration,
                    'Commentaire': comment
                })

            # Bouton pour soumettre l'analyse
            submitted = st.form_submit_button("Soumettre analyse")

            if submitted:
                # Créer un DataFrame à partir des données d'analyse
                analysis_df = pd.DataFrame(analysis_data)

                # Stocker les données d'analyse dans la base de données SQLite
                for data in analysis_data:
                    c.execute('''INSERT INTO analysis_data
                                 (conversation_id, generated_response, respect_context, 
                                 completion, good_practices, no_aberration, comment)
                                 VALUES (?, ?, ?, ?, ?, ?, ?)''',
                              (st.session_state.current_conversation["id"], 
                               data['generated_response'], 
                               data['Respect du contexte'],
                               data['Complétion de la réponse'], 
                               data['Respect des bonnes pratiques'], 
                               data['Absence d’aberration'],
                               data['Commentaire']))
                conn.commit()

                st.session_state.analysis_inputs = {}

                st.session_state["form"] = None

                # Réinitialiser les valeurs
                for idx in range(1, len(st.session_state.current_response) + 1):
                    st.session_state.analysis_inputs[f"context_{idx}"] = ""
                    st.session_state.analysis_inputs[f"completion_{idx}"] = ""
                    st.session_state.analysis_inputs[f"practices_{idx}"] = ""
                    st.session_state.analysis_inputs[f"aberration_{idx}"] = ""
                    st.session_state.analysis_inputs[f"comment_{idx}"] = ""

                st.success("Analyse soumise avec succès!")



                

    else:
        st.write("Aucune réponse à analyser pour le moment.")

# Fermer la connexion à la base de données à la fin de l'exécution
#conn.close()