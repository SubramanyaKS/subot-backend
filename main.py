from fastapi import FastAPI, Query,Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from service.llm import llm_call
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from service.save import save_chunk
from utils.constant import origins,url

load_dotenv()



app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"}
))

class QueryInput(BaseModel):
    query: str


vectorstore= save_chunk(url)


@app.post("/ask")
@limiter.limit("2/minute") 
async def ask(input: QueryInput,request:Request):
    print("Called")
    query = input.query
    # Initialize Gemini model
    response = llm_call(query,vectorstore)
    
    return {"response": response.content}
