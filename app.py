import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
#from InstructorEmbedding import INSTRUCTOR
#from angle_emb import AnglE
from langchain_openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import faiss
import voyageai
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def getTextChunks(raw_text):
    textsplitter = CharacterTextSplitter(
        separator='\n' ,
        chunk_size = 1000 ,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = textsplitter.split_text(raw_text)
    return chunks


import numpy as np

def get_vectorstore(text_chunks):
    vo = voyageai.Client(api_key="pa-Irenis7B3cGCAVf9D45-GMcoSCNXkuEGvSa8hrVLBOE")
    embeddings = vo.embed(texts = text_chunks, model = 'voyage-2')
    def embedding_function(text):
        return vo.embed(texts=[text], model="voyage-2").embeddings[0]
    # Create FAISS vector store directly from embeddings
    TE = zip(text_chunks, embeddings.embeddings)
    vectorstore = FAISS.from_embeddings(text_embeddings=TE, embedding = embedding_function)  # Specify dimensionality
     # Add embeddings to the index
    
    return vectorstore

def get_conversation_chain(vectorstore):
    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory = memory

    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title = "Chat with multiple PDFs", page_icon = ":books:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    
    st.header("Chat with multiple PDFs :books:")
    


    user_question = st.text_input("Ask questions from your pdf here:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("upload your PDFs here and clickk on Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = getTextChunks(raw_text)
                #   st.write(text_chunks)
                
                vectorstore = get_vectorstore(text_chunks)
                #st.write(embeddings)
                st.write('Process completed.')
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                


if __name__ == '__main__':
    main()






