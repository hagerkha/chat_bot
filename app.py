from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client
import os

app = Flask(__name__)
CORS(app)

# استخدام Hugging Face Space
client = Client("alim9hamed/medical_chatbot", src="spaces")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        # استدعاء النموذج من Gradio Space
        result = client.predict(
            question,
            api_name="/predict"
        )

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # استخدام PORT من متغيرات البيئة (علشان Railway يشتغل صح)
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
