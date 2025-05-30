from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import os

app = Flask(__name__)
CORS(app)

# تهيئة gradio_client مع دعم مفتاح API (اختياري)
try:
    client = Client(
        "https://alim9hamed-medical-chatbot.hf.space",
        hf_token=os.environ.get("HUGGINGFACE_API_TOKEN", None)
    )
    print("Successfully initialized gradio_client")
except Exception as e:
    print(f"Failed to initialize gradio_client: {str(e)}")
    raise

@app.route('/')
def home():
    return jsonify({"message": "Server is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        print(f"Received question: {question}")
        result = client.predict(
            question,
            api_name="/predict"
        )
        print(f"Prediction result: {result}")

        return jsonify({"response": result})

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)