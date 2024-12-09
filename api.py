from flask import Flask, request, jsonify
import subprocess
from TextProcessing.process_text import process_inputs, evaluate_responses
from DataMigration.database_functions import create_tables, log_to_sqlite
import os

app = Flask(__name__)

# Route pour servir la page Streamlit
@app.route('/', methods=['GET'])
def serve_streamlit():
    try:
        # Exécuter la commande pour démarrer Streamlit
        streamlit_script_path = os.path.join(os.getcwd(), 'app.py')  # Utilise le bon chemin pour ton fichier 'app.py'
        subprocess.Popen(["python3","-m","streamlit", "run", streamlit_script_path, "--server.port=5001"])
        return "Streamlit is running. Please check your browser.", 200
    except Exception as e:
        return jsonify({"error": f"Failed to start Streamlit: {str(e)}"}), 500

# Route pour traiter le texte
@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text')
    dropout_prob = data.get('dropout_prob', 0)
    num_samples = data.get('num_samples', 1)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        results_df = process_inputs(text, dropout_prob, num_samples)
        return jsonify(results_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": f"Error in processing input: {str(e)}"}), 500

# Route pour évaluer les réponses
@app.route('/evaluate_responses', methods=['POST'])
def evaluate_responses_endpoint():
    data = request.json
    GT1 = data.get('GT1')
    responses = data.get('responses')

    if not GT1 or not responses:
        return jsonify({"error": "GT1 and responses are required"}), 400

    try:
        final_df = evaluate_responses(GT1, responses)
        return jsonify(final_df.to_dict(orient='records')), 200
    except Exception as e:
        return jsonify({"error": f"Error in evaluating responses: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Error starting Flask app: {str(e)}")