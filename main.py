from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langdetect import detect, DetectorFactory
import os

from fastapi.middleware.cors import CORSMiddleware

# إعداد التطبيق
app = FastAPI()

# تفعيل CORS لو هتستخدمي من Flutter أو من متصفح
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DetectorFactory.seed = 0

class QuestionRequest(BaseModel):
    question: str

class EyeDiseaseChatbot:
    def __init__(self):
        self.llm = ChatGroq(
            api_key="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", 
            model_name="deepseek-r1-distill-qwen-32b"
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        self.vector_db = self.initialize_vector_db()

    def initialize_vector_db(self):
        sites = [
            "https://www.webmd.com/eye-health/",
            "https://www.mayoclinic.org/diseases-conditions/index?letter=E",
            "https://medlineplus.gov/eyeandvision.html",
            "https://www.healthline.com/health/eye-health",
            "https://www.cdc.gov/visionhealth/",
        ]
        all_docs = []
        for site in sites:
            try:
                loader = WebBaseLoader(site)
                docs = loader.load()
                all_docs.extend(docs)
            except:
                continue

        if not all_docs:
            from langchain.docstore.document import Document
            all_docs = [Document(page_content="Basic eye health information", metadata={"source": "fallback"})]

        chunks = self.text_splitter.split_documents(all_docs)
        return Chroma.from_documents(chunks, self.embedding_model)

    def generate_answer(self, question):
        try:
            language = detect(question)
        except:
            language = "en"

        similar_docs = self.vector_db.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in similar_docs]) if similar_docs else ""

        template = """You are an ophthalmologist...
        {context}
        {question}
        Answer:"""
        
        prompt = PromptTemplate(template=template, input_variables=['context', 'question', 'language'])
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        if not context:
            return "No answer is currently available." if language == "en" else "لا يوجد إجابة متاحة حاليًا."

        result = chain({
            "input_documents": similar_docs,
            "question": question,
            "language": language
        })

        return result["output_text"]

# تهيئة البوت
chatbot = EyeDiseaseChatbot()

@app.get("/")
def home():
    return {"message": "Eye Disease Chatbot API is running!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = chatbot.generate_answer(request.question)
    return {"answer": answer}
