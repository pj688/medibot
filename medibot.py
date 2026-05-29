#streamlit run medibot.py
#ctrl c for stopping streamlit
import os 
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db 

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = ["context","question"])
    return prompt



def main():
    st.title("ASK CHATBOT!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    prompt = st.chat_input("PASS YOUR PROMPT HERE")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("FAILED TO LOAD THE VECTOR STORE")

            GROQ_MODEL_NAME = "llama-3.1-8b-instant" #change to any supoorted groq model
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            llm = ChatGroq(
    model = GROQ_MODEL_NAME,
    temperature = 0.5,
    max_tokens = 512,
    api_key = GROQ_API_KEY,
)

            # STEP 3: BUILD RAG CHAIN
            retrieval_qa_Chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            # DOCUMENT COMBINER CHAIN (STUFF DOCUMENTS INTO PROMPT)
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_Chat_prompt)

            # RETRIEVAL CHAIN (RETRIEVER + DOC COMBINER)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

            #NOW INVOKE WITH A SINGLE QUERY
            # user_query = input("Write Query Here: ")
            response = rag_chain.invoke({'input': prompt})

            result = response["answer"]
            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({'role':'assistant','content':result})

        except Exception as e:
            st.error(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()