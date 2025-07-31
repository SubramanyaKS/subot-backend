from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_MODEL = os.getenv("GEMINI_API_MODEL", "gemini-1.0-flash")
GEMINI_API_EMBEDDING_MODEL = os.getenv("GEMINI_API_EMBEDDING_MODEL")
USER_AGENT=os.getenv('USER_AGENT')