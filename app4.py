import os
import json
import boto3
import numpy as np
import streamlit as st
from operator import itemgetter
import time
import glob
import tempfile
import shutil

# Disable LangChain tracing to avoid TracerException
# os.environ["LANGCHAIN_TRACING_V2"] = "false"

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

def ensure_data_directory():
    """Ensure the data directory exists"""
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        st.info(f"Created {data_path} directory")

def save_uploaded_files(uploaded_files):
    """Save uploaded files to the data directory"""
    ensure_data_directory()
    saved_files = []
    
    for uploaded_file in uploaded_files:
        # Create a safe filename
        filename = uploaded_file.name
        filepath = os.path.join(data_path, filename)
        
        # Save the file
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        saved_files.append(filename)
        
    return saved_files

def get_existing_files():
    """Get list of existing PDF files in data directory"""
    if not os.path.exists(data_path):
        return []
    
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    return [os.path.basename(f) for f in pdf_files]

def delete_file(filename):
    """Delete a specific file from data directory"""
    filepath = os.path.join(data_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False

def ingest_data(data_path):
    import glob
    
    # Count actual PDF files
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    num_pdf_files = len(pdf_files)
    
    loader = PyPDFDirectoryLoader(data_path)
    documents = loader.load()  # This loads each PAGE as a separate document
    
    if len(documents) == 0:
        st.warning(f"No PDF files found in {data_path} directory")
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    # More informative logging
    st.info(f"📄 Found {num_pdf_files} PDF file(s)")
    st.info(f"📑 Loaded {len(documents)} pages total")
    st.info(f"✂️ Split into {len(docs)} text chunks")
    
    return docs

def store_vectors(docs):
    if len(docs) == 0:
        st.error("No documents to vectorize!")
        return False
    
    vector_store = FAISS.from_documents(
        docs,
        embeddings
    )
    vector_store.save_local("faiss_index")
    return True

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

@st.cache_resource
def build_chat_rag_chain(_llm, _vector_store):
    """Cache the RAG chain to avoid rebuilding on every interaction"""
    retriever = _vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={
            'k': 5,  
            'fetch_k': 10 
        }
    )
    
    history_aware_retriever = create_history_aware_retriever(_llm, retriever, contextualize_Q_prompt)
    QA_chain = create_stuff_documents_chain(llm=_llm, prompt=QA_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, QA_chain)

    rag_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return rag_with_memory

def display_chat_message(role, message, timestamp=None):
    """Enhanced chat message display with timestamps"""
    with st.chat_message(role):
        st.markdown(message)
        if timestamp and st.session_state.get('show_timestamps', False):
            st.caption(f"*{timestamp}*")

def main():
    st.set_page_config(page_title='RAG Chat PDF', layout='wide', initial_sidebar_state='expanded')
    
    # Header with some styling
    st.title('🤖 Chat with your Knowledge Base')
    st.markdown('*Powered by AWS Bedrock LLM*')
    st.divider()

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session-" + os.urandom(4).hex()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load vector store
    try:
        index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        st.success("✅ Vector store loaded successfully!")
        vector_count = index.index.ntotal if hasattr(index.index, 'ntotal') else "Unknown"
        st.info(f"📊 Vector store contains: {vector_count} vectors")
    except Exception as e:
        st.error(f"❌ Vector store not found: {e}")
        st.info("👈 Use the sidebar to create one.")
        index = None

    # Sidebar
    with st.sidebar:
        st.header('Configuration : ')
        
        # File Upload Section
        st.subheader('Upload Documents')
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload one or more PDF documents to add to your knowledge base"
        )
        
        if uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button('💾 Save Files', use_container_width=True):
                    with st.spinner('Saving files...'):
                        saved_files = save_uploaded_files(uploaded_files)
                        st.success(f"✅ Saved {len(saved_files)} file(s)!")
                        for filename in saved_files:
                            st.write(f"- {filename}")
                        time.sleep(1)
                        st.rerun()
            
            with col2:
                if st.button('🔄 Save & Process', use_container_width=True):
                    with st.spinner('Saving and processing files...'):
                        # Save files first
                        saved_files = save_uploaded_files(uploaded_files)
                        st.success(f"✅ Saved {len(saved_files)} file(s)!")
                        
                        # Then process all files in directory
                        try:
                            docs = ingest_data(data_path)
                            if len(docs) > 0:
                                success = store_vectors(docs)
                                if success:
                                    st.success(f"🎉 Vector store updated!")
                                    time.sleep(1)
                                    st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error processing: {e}")
        
        st.divider()
        
        # Existing Files Management
        st.subheader('Existing Files')
        existing_files = get_existing_files()
        
        if existing_files:
            st.write(f"**{len(existing_files)} file(s) in knowledge base:**")
            
            # Show files with delete option
            for filename in existing_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {filename}")
                with col2:
                    if st.button('🗑️', key=f"del_{filename}", help=f"Delete {filename}"):
                        if delete_file(filename):
                            st.success(f"Deleted {filename}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to delete {filename}")
        else:
            st.info("No PDF files found. Upload some documents above!")
        
        st.divider()
        
        st.subheader('Vector Store Management')
        if st.button('🔄 Rebuild Vector Store', use_container_width=True):
            if not existing_files:
                st.warning("No files to process! Upload some PDFs first.")
            else:
                with st.spinner('📚 Processing all documents...'):
                    try:
                        docs = ingest_data(data_path)
                        if len(docs) > 0:
                            success = store_vectors(docs)
                            if success:
                                st.success(f"✅ Vector store rebuilt with {len(docs)} chunks!")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {e}")
        
        st.divider()
        
        # Settings
        st.subheader('Settings')
        st.session_state.show_timestamps = st.checkbox('Show timestamps', value=False)
        
        # Model parameters (for advanced users)
        with st.expander('🔬 Advanced Model Settings'):
            st.info('Current model: meta.llama3-8b-instruct-v1:0')
            st.slider('Temperature', 0.0, 1.0, 0.1, disabled=True, help='Currently fixed at 0.1')
            st.slider('Top P', 0.0, 1.0, 0.9, disabled=True, help='Currently fixed at 0.9')
            st.number_input('Max Tokens', min_value=100, max_value=1000, value=512, disabled=True)
        
        st.divider()
        
        # Chat controls
        st.subheader('Chat Controls')
        if st.button('🗑️ Clear Conversation', use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.session_id in store:
                store[st.session_state.session_id] = ChatMessageHistory()
            st.success('Conversation cleared!')
            time.sleep(0.5)
            st.rerun()

    # Main chat interface
    if index:
        llm = get_LLM(LLM_model_id)
        chat_rag_chain = build_chat_rag_chain(llm, index)
        
        # Chat input
        question = st.chat_input('💭 Ask a question about your documents...')

        if question:
            # Add user message immediately
            timestamp = time.strftime("%H:%M:%S")
            st.session_state.chat_history.append(('user', question, timestamp))
            
            with st.spinner('🤔 Generating response...'):
                try:
                    start_time = time.time()
                    response = chat_rag_chain.invoke(
                        {"input": question},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    end_time = time.time()
                    
                    answer = response['answer']
                    response_time = f"{end_time - start_time:.2f}s"
                    timestamp = time.strftime("%H:%M:%S")
                    
                    st.session_state.chat_history.append(('assistant', answer, timestamp))
                    
                    # Show response time in sidebar for debugging
                    st.sidebar.success(f'⚡ Response generated in {response_time}')
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")
                    st.info("🔄 Please try again or check your configuration.")
        
        # Display chat history
        for i, chat_item in enumerate(st.session_state.chat_history):
            role = chat_item[0]
            message = chat_item[1]
            timestamp = chat_item[2] if len(chat_item) > 2 else None
            
            display_chat_message(role, message, timestamp)

        # Show helpful tips when no conversation
        if len(st.session_state.chat_history) == 0:
            st.info("""
            👋 **Welcome!** Here are some tips to get started:
            
            - Ask specific questions about your documents
            - Try follow-up questions to dive deeper into topics
            - Use clear, focused questions for better results
            - The system remembers our conversation context

            """)
    
    else:
        st.warning("""
        📁 **No vector store found!** 
        
        To get started:
        1. **Upload PDF files** using the sidebar file uploader
        2. Click "💾 Save Files" or "🔄 Save & Process"
        3. Wait for processing to complete
        4. Start chatting with your documents!
        
        **Alternative:** 
        - Add PDF files manually to the `data/` directory
        - Click "🔄 Rebuild Vector Store" in the sidebar
        """)

if __name__ == "__main__":
    main()