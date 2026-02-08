import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from chat_ui import css, bot_template, user_template
import boto3
import os
import time
from operator import itemgetter


def save_file_to_s3(file):
    s3 = boto3.client('s3', aws_access_key_id=os.getenv("aws_access_key"), aws_secret_access_key=os.getenv("aws_secret_access_key"))
    file.seek(0)
    s3.upload_fileobj(file, os.getenv("aws_bucket"), file.name)

def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    """Create vector store with retry logic for rate limits."""
    # Using HuggingFace embeddings - free and no API key needed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vector_store
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15  # 15, 30, 45 seconds
                    st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    st.error("❌ **API Quota Exceeded!**")
                    st.error("Your Google API key has reached its quota limit. Please:")
                    st.markdown("""
                    1. **Wait**: Quotas reset at midnight Pacific Time
                    2. **Check Usage**: Visit https://ai.dev/rate-limit
                    3. **Create New Key**: Get a new API key at https://makersuite.google.com/app/apikey
                    4. **Enable Billing**: For higher quotas, enable billing in Google Cloud Console
                    """)
                    raise
            else:
                raise

def get_conversation_chain(vector_store):
    # Using gemini-2.5-flash - available in API key with 5 RPM limit
    # This model is newer and should have fresh quotas
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", convert_system_message_to_human=True)
    retriever = vector_store.as_retriever()
    
    # Helper function to format documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create a prompt template for conversational retrieval
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    # Create the chain with proper input extraction
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever

def handle_user_question(user_question):
    st.write("User Question: ", user_question)
    
    # Get the chain and retriever
    chain = st.session_state.conversation
    retriever = st.session_state.retriever
    
    try:
        # Get relevant documents using modern invoke() method
        source_documents = retriever.invoke(user_question)
        
        # Invoke the chain with chat history - with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                answer = chain.invoke({
                    "question": user_question,
                    "chat_history": st.session_state.chat_history
                })
                break  # Success, exit retry loop
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 5  # Wait 5 seconds before retry
                        st.warning(f"⏳ Rate limit hit. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error("❌ **API Quota Exhausted!**")
                        st.error("Your Google API key has reached its quota limit.")
                        st.markdown("""\n**Solutions:**
                        1. ⏰ **Wait**: Quotas reset at midnight Pacific Time
                        2. 🔍 **Check Usage**: https://ai.dev/rate-limit
                        3. 🔑 **New API Key**: https://makersuite.google.com/app/apikey
                        4. 💳 **Enable Billing**: For much higher quotas
                        5. 🕐 **Try Later**: Free tier has daily limits
                        """)
                        return  # Exit function without proceeding
                else:
                    raise  # Re-raise non-quota errors
        
        print("Response: ", answer)
        
        # Store in session state
        st.session_state.answer = answer
        st.session_state.source_documents = source_documents
        
        # Add to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # Log the response
        st.write("LLM Response: ", answer)
        
        # Display source documents inside a collapsible expander
        if source_documents:
            with st.expander("Source Documents (Click to expand)"):
                for i, doc in enumerate(source_documents):
                    st.write(f"Document {i+1}:")
                    st.write(doc.page_content)
                    st.write("---")
        
        # Display the chat history with alternating user and bot messages
        for i, message in enumerate(st.session_state.chat_history):
            if isinstance(message, HumanMessage):
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Try uploading a smaller PDF or creating a new API key if quotas are exhausted.")

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Data analyzer - using RAG.", page_icon=":books:")

    # Apply the custom CSS from chat_ui.py
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header and input area for user question
    st.header("PDF Data analyzer - using RAG.")
    user_question = st.text_input("Ask a question about the inputted PDFs:")

    if user_question:
        handle_user_question(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.subheader("Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs is not None:
                with st.spinner("Processing"):
                    raw_text = ""
                    for pdf in pdf_docs:
                        raw_text += get_pdf_text(pdf)

                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)

                    chain, retriever = get_conversation_chain(vector_store)
                    st.session_state.conversation = chain
                    st.session_state.retriever = retriever
                    st.session_state.chat_history = []  # Reset chat history
                    print("Chain created successfully")

if __name__ == '__main__':
    main()
