from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from utils.parameter import GEMINI_API_KEY, GEMINI_API_MODEL

def llm_call(query,vectorstore):
    llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY,model=GEMINI_API_MODEL, temperature=0.7)

    # Define your RAG prompt
# "Context:\n{context}\n\n"
    # "Question: {query}\n"
    # "Answer briefly, only the key points. Do not elaborate."

    # template = """
    # Context: {context}
    # Question: {query}
    # Be brief. Only list key facts.
    # """

    template = """
    You are a chatbot assistant for Subramanya's personal portfolio website..
Context: {context}
User Question: {query}

Instructions:
1. If the User Question is a simple greeting (e.g., "Hi", "Hello", "Hey", "Good morning", "What's up?"), respond with a friendly greeting and offer to help. Do NOT provide any portfolio details or context information immediately.
2. If the User Question is too general or unclear, ask for more specific details about what they'd like to know. Do NOT provide all portfolio details.
3. Otherwise, answer the question using only the information in the context provided.
4. Only list key facts relevant to the user's question.
5. If the requested information is not in the context, politely state that you cannot find it.
6. Keep the tone friendly, concise, and professional.
"""
    
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever()
    
    # Retrieve relevant documents
    docs = retriever.invoke(query)
    
    # Generate response
    response = llm.invoke(prompt.format(context=docs, query=query))

    return response
