import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# STEP 1: SETUP GROQ LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant" #change to any supoorted groq model
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

llm = ChatGroq(
    model = GROQ_MODEL_NAME,
    temperature = 0.5,
    max_tokens = 512,
    api_key = GROQ_API_KEY,
)

# STEP 2: CONNECT LLM WITH FAISS AND CREATE CHAIN

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context","question"])
    return prompt

# LOAD DATABASE
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model,allow_dangerous_deserialization=True)

# STEP 3: BUILD RAG CHAIN
retrieval_qa_Chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# DOCUMENT COMBINER CHAIN (STUFF DOCUMENTS INTO PROMPT)
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_Chat_prompt)

# RETRIEVAL CHAIN (RETRIEVER + DOC COMBINER)
rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)


#NOW INVOKE WITH A SINGLE QUERY
user_query = input("Write Query Here: ")
response = rag_chain.invoke({'input': user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS: ")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
    