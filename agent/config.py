import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("GEMINI API:", GEMINI_API_KEY)
print("GROQ API:", GROQ_API_KEY)
print("Working Directory:", os.getcwd())