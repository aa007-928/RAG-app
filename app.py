import boto3
import json
import os
import numpy as np
import streamlit as st

from langchain_aws import BedrockEmbeddings

from langchain_aws.llms.bedrock import BedrockLLM

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
#from langchain.chains import retrieval_qa
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain.chains import create_retrieval_chain 
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


data_path = "data"

embed_model_id = "amazon.titan-embed-text-v2:0"
LLM_model_id = "meta.llama3-8b-instruct-v1:0"

bedrock = boto3.client('bedrock-runtime',region_name='us-east-1')
embeddings = BedrockEmbeddings(model_id = embed_model_id,client=bedrock)

def ingest_data(data_path):
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def store_vectors(docs):
    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )
    vector_store.save_local("faiss_index")

def get_LLM(LLM_model_id):
    llm = BedrockLLM(model_id=LLM_model_id,client=bedrock,model_kwargs={'max_gen_len':512})
    return llm


prompt_template ="""
You are an AI assistant helping to answer questions based on the provided context.
Answer the following question accurately and concisely using only the information in the context. Give the answer in approximately 250 words. If the answer is not present in the context, respond with "I don't know."
<context>
{context}
</context>

Question:{question}
Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template,input_variables=["context","question"])


def generate_response(llm,vector_store,query):
    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':3})
    stuff_chain = create_stuff_documents_chain(llm=llm,prompt=PROMPT)
    rag_chain = RunnableParallel({
        "context" : retriever,
        "question": RunnablePassthrough()
    }) | stuff_chain

    response = rag_chain.invoke(query)
    print(response)
    return response


def main():
    st.set_page_config('Chat PDF')
    st.header('Chat with your Knowledge Base using LLM from AWS Bedrock')

    question = st.text_input('Ask question to your Knowlwdge Base')

    with st.sidebar:
        st.title('Update or Create Vector Store')
        if st.button('Update Vector Store'):
            with st.spinner('Processing....'):
                docs = ingest_data(data_path)
                store_vectors(docs)
                st.success("Done")
        
    if st.button('Generate Output'):
        with st.spinner('Processing....'):
            index = FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)
            llm = get_LLM(LLM_model_id)
            
            st.write(generate_response(llm,index,question))
            st.success('Done')


if __name__ == "__main__":
    main()




