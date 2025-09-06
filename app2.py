import os
import json
import boto3
import numpy as np
import streamlit as st
import time

from langchain_aws import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

data_path = "data"

embed_model_id = "amazon.titan-embed-text-v2:0"
LLM_model_id = "meta.llama3-8b-instruct-v1:0"

bedrock = boto3.client('bedrock-runtime',region_name='us-east-1')
embeddings = BedrockEmbeddings(model_id = embed_model_id,client=bedrock)

def ingest_data(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    return docs

def store_vectors(docs):
    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )
    vector_store.save_local("faiss_index")

def get_LLM(LLM_model_id):
    llm = BedrockLLM(
        model_id=LLM_model_id,
        client=bedrock,
        model_kwargs={
            'max_gen_len': 512,
            'temperature': 0.1, 
            'top_p': 0.9
        }
    )
    return llm

# contextualization prompt - give specific instructions
contextualize_Q_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
        
        Examples:
        - If user asks "What about its architecture?" after discussing SuperGlue, reformulate to "What is the architecture of SuperGlue?"
        - If user asks "Can you tell me more about it?" after discussing a topic, reformulate to include the specific topic
        - If the question is already standalone, return it unchanged"""
    ),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("Question: {input}")
])

# QA prompt - give clearer instructions and better formatting
QA_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a helpful AI assistant answering questions based on the provided context. 
        
        Instructions:
        1. Use ONLY the information provided in the context below to answer questions
        2. If the context contains relevant information, provide a clear and helpful answer
        3. If the context does NOT contain enough information to answer the question, respond with "I don't have enough information in the provided context to answer that question."
        4. Do not make up information or use knowledge outside of the provided context
        5. Keep your answer concise but comprehensive (around 200-300 words)
        6. Reference specific details from the context when possible
        
        Context: {context}"""
    ),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template("Human: {input}\n\nAssistant:")
])

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    history = store[session_id]
    
    window_size = 10  # sliding window for context history
    if len(history.messages) > window_size:
        history.messages = history.messages[-window_size:]
    
    return history

def build_chat_rag_chain(llm, vector_store):
    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 5,  
            'fetch_k': 10 
        }
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_Q_prompt)
    QA_chain = create_stuff_documents_chain(llm=llm, prompt=QA_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, QA_chain)

    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return rag_with_memory

def main():
    st.set_page_config(page_title='Chat PDF', layout='wide')
    st.header('Chat with your Knowledge Base using LLM from AWS Bedrock')

    try:
        index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.success("Vector store loaded successfully!")
    except Exception as e:
        st.error(f"Vector store not found: {e}")
        st.info("Use the sidebar to create one.")
        index = None

    with st.sidebar:
        st.title('Update or Create Vector Store')
        if st.button('Update Vector Store'):
            with st.spinner('Processing documents...'):
                try:
                    docs = ingest_data(data_path)
                    if len(docs) == 0:
                        st.error("No documents found in the data directory!")
                    else:
                        store_vectors(docs)
                        st.success(f"Vector store created with {len(docs)} document chunks!")
                        time.sleep(1)  # Brief pause for user to see success message
                        st.rerun()  # Refresh to load the new index
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        
        #some debugging info
        # if index:
        #     st.write(f"Vector store contains: {index.index.ntotal} vectors")
    
    if index:
        llm = get_LLM(LLM_model_id)
        chat_rag_chain = build_chat_rag_chain(llm, index)
        
        if "session_id" not in st.session_state:
            st.session_state.session_id = "session-" + os.urandom(4).hex()
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # clear conversation button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                if st.session_state.session_id in store:
                    store[st.session_state.session_id] = ChatMessageHistory()
                st.rerun()

        question = st.chat_input('Ask a question about your documents...')

        if question:
            with st.spinner('Generating response...'):
                try:
                    response = chat_rag_chain.invoke(
                        {"input": question},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    
                    answer = response['answer']
                    # context_docs = response.get('context', [])
                    
                    st.session_state.chat_history.append(('user', question))
                    st.session_state.chat_history.append(('assistant', answer))
                    
                    # Show retrieved context in expander for debugging
                    # if context_docs and st.sidebar.checkbox("Show Retrieved Context (Debug)"):
                    #     with st.expander("Retrieved Context"):
                    #         for i, doc in enumerate(context_docs):
                    #             st.write(f"**Chunk {i+1}:**")
                    #             st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    #             st.write("---")
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.write("Please try again or check your configuration.")
        
        # Display chat history
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

if __name__ == "__main__":
    main()


