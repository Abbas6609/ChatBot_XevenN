# Necessary imports for Application
import streamlit as st
import streamlit as st
import os
from PyPDF2 import PdfReader
import pinecone
import shutil
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Qdrant        
import random
from datetime import datetime
from langchain.chains import RetrievalQA
import string
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
from langchain.docstore.document import Document
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
import zipfile
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


# Credentials
openapi_key = st.secrets["OPENAI_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone_env = st.secrets["PINECONE_ENV"]

# Main function to run the Streamlit ap
def main():
# Load environment variables
    load_dotenv()
    st.set_page_config(page_title="Q/A with your file")
    st.header("Hackathon Chatbot Application")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'zip'], accept_multiple_files=True)
        vector_store_option = st.radio("Choose your Vector Store", ('Qdrant', 'Pinecone'))
        embeddings_model = st.radio("Choose Embeddings Model", ("intfloat/e5-large-v2", "BAAI/bge-small-en-v1.5"))
        text_splitter_option = st.radio("Choose Text Splitting Method", ('CharacterTextSplitter', 'RecursiveCharacterTextSplitter'))
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # Select embeddings model
        if embeddings_model == "BAAI/bge-small-en-v1.5":
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

        text_chunks_list = []
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = os.path.splitext(file_name)[1].lower()

            if file_extension == ".zip":
                extracted_files = extract_files_from_zip(uploaded_file)
                for ext_file_name, ext_file_path in extracted_files:
                    file_text = get_files_text(ext_file_path)
                    text_chunks = get_text_chunks(file_text, ext_file_name, text_splitter_option)
                    text_chunks_list.extend(text_chunks)
            else:
                use_pymupdf = True  # Default for PDFs
                use_docx2txt = True  # Default for DOCX files
                file_text = get_files_text(uploaded_file, use_pymupdf, use_docx2txt)
                text_chunks = get_text_chunks(file_text, file_name, text_splitter_option)
                text_chunks_list.extend(text_chunks)

        curr_date = str(datetime.now())
        collection_name = "".join(random.choices(string.ascii_letters, k=4)) + curr_date.split('.')[0].replace(':', '-').replace(" ", 'T')
            
        # Create vector store only if embeddings model is selected
        if embeddings is not None:
            vectorestore = get_vectorstore(text_chunks_list, collection_name, embeddings, vector_store_option)
            if vectorestore is not None:
                st.session_state.conversation = get_qa_chain(vectorestore, num_chunks=4)
                st.session_state.processComplete = True
                st.write(f"Vector Store ({vector_store_option}) and QA Chain Created.")
        else:
            st.error("Embeddings model is not selected or initialized.")

    if st.session_state.processComplete:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

# Function to extract files from a zip archive
def extract_files_from_zip(uploaded_zip_file):
    extracted_files = []
    tmp_dir = tempfile.mkdtemp()  # Change to mkdtemp to keep the directory
    try:
        with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                if file.endswith('.pdf') or file.endswith('.docx'):
                    file_path = os.path.join(root, file)
                    extracted_files.append((file, file_path))
    finally:
        pass
    return extracted_files

def get_pdf_text(file_path):
    # Directly using file_path to read the PDF
    reader = PdfReader(file_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text() is not None])
    return text

# Function to get text from a PDF file using PyMuPDF
def get_pdf_text_pymupdf(file_path):
    # Process the PDF file using PyMuPDFLoader with a file path
    loader = PyMuPDFLoader(file_path)
    data = loader.load()
    text = "\n".join([doc.page_content for doc in data])
    return text

# Function to get text from various file formats
def get_files_text(file_input, use_pymupdf=False, use_docx2txt=False):
    text = ""

    # Temporarily save the uploaded file if it's not a file path
    if not isinstance(file_input, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_input.name)[1]) as tmp_file:
            shutil.copyfileobj(file_input, tmp_file)
            file_path = tmp_file.name
    else:
        file_path = file_input

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        if use_pymupdf:
            text += get_pdf_text_pymupdf(file_path)
        else:
            text += get_pdf_text(file_path)
    elif file_extension == ".docx":
        if use_docx2txt:
            text += get_docx_text_docx2txt(file_path)
        else:
            text += get_docx_text_unstructured(file_path)

    # Remove the temporarily saved file
    if not isinstance(file_input, str):
        os.remove(file_path)

    return text


def get_docx_text_docx2txt(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    loader = Docx2txtLoader(tmp_file_path)
    data = loader.load()

    text = "\n".join([doc.page_content for doc in data])

    os.remove(tmp_file_path)
    return text

def get_docx_text_unstructured(file_path):
    # Directly using file_path to load the DOCX file
    loader = UnstructuredWordDocumentLoader(file_path)
    data = loader.load()

    text = "\n".join([doc.page_content for doc in data])
    return text

# Function to get text chunks based on user's choice of text splitter
def get_text_chunks(text, filename, text_splitter_option):
    if text_splitter_option == 'RecursiveCharacterTextSplitter':
        return get_recursive_text_chunks(text, filename)
    elif text_splitter_option == 'CharacterTextSplitter':
        return get_character_text_chunks(text, filename)
    else:
        st.error("Invalid text splitter option")
        return []

def get_recursive_text_chunks(text, filename):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return create_document(text, filename, text_splitter)

def get_character_text_chunks(text, filename):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return create_document(text, filename, text_splitter)

# Function to create document objects from text chunks
def create_document(text, filename, text_splitter):
    chunks = text_splitter.split_text(text)
    doc_list = []
    for i, chunk in enumerate(chunks):
        metadata = {"source": f"{filename}_{i + 1}"}
        doc_string = Document(page_content=chunk, metadata=metadata)
        doc_list.append(doc_string)
    return doc_list

# Function to get the Pinecone or Qdrant vector store based on user's choice
def get_vectorstore(text_chunks, COLLECTION_NAME, embeddings, vector_store_option):
    if vector_store_option == "Qdrant":
        try:
            knowledge_base = Qdrant.from_documents(
                documents=text_chunks,
                embedding=embeddings,
                url=qdrant_url,
                prefer_grpc=True,
                api_key=qdrant_api_key,
                collection_name=COLLECTION_NAME,
            )
        except Exception as e:
            st.error(f"Error in creating Qdrant vector store: {e}")
            return None

    elif vector_store_option == "Pinecone":
        try:
            pinecone_api_key = st.secrets["PINECONE_API_KEY"]
            pinecone_env = st.secrets["PINECONE_ENV"]
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
            knowledge_base = Pinecone.from_documents(
                documents=text_chunks, 
                embedding=embeddings, 
                index_name="bot-e5-large"
            )
        except Exception as e:
            st.error(f"Error in creating Pinecone vector store: {e}")
            return None
    return knowledge_base

# Function to create the QA chain using LangChain's RetrievalQA
def get_qa_chain(vectorstore,num_chunks):
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model = "gpt-3.5-turbo-16k"), chain_type="stuff",
                                retriever=vectorstore.as_retriever(search_type="similarity",
                                                            search_kwargs={"k": num_chunks}),  return_source_documents=True)
    return qa

# Function to handle user input and generate responses
def handel_userinput(user_question):
    with st.spinner('Generating response...'):
        result = st.session_state.conversation({"query": user_question})
        response = result['result']

        # Check if source documents are available
        if result['source_documents']:
            source = result['source_documents'][0].metadata['source']
            response_message = f"{response} \n Source Document: {source}"
        else:
            response_message = response  # Or some default message

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(response_message)

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))


if __name__ == '__main__':
    main()