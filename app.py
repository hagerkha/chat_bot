from flask import Flask, request, jsonify
from flask_cors import CORS
from gradio_client import Client

app = Flask(__name__)
CORS(app)  # السماح للطلبات من مصادر خارجية (مثل تطبيق Flutter)

# تحميل الـ Client الخاص بـ Hugging Face
client = Client("alim9hamed/medical_chatbot")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # استلام البيانات بصيغة JSON
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        # استدعاء نموذج الـ Chatbot من Hugging Face
        result = client.predict(
            question,  # تعديل طريقة تمرير السؤال
            api_name="/predict"
        )

        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
