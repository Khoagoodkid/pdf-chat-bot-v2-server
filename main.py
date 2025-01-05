from typing import Union
from langchain.schema import Document
from fastapi import FastAPI, File, UploadFile, Form
import chromadb
from langchain_community.document_loaders import PyMuPDFLoader
import uuid 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import base64
from openai import OpenAI
from chromadb.utils import embedding_functions
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import boto3
import os 
from dotenv import load_dotenv
import pdfplumber
import io

load_dotenv()



OPEN_API_KEY = os.getenv("OPEN_API_KEY") 
open_ai_client = OpenAI(api_key=OPEN_API_KEY)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = chromadb.HttpClient(host='54.196.133.247', port=8000)
# client.heartbeat()
s3 = boto3.client("s3")
BUCKET_NAME = "pdf-chat-bot-djn232db"

@app.get("/")
def read_root():
    return {"Hello": "World"}



def upload_to_s3_bucket(file: UploadFile = File(...)):
    try:
        # Upload the file to S3
        s3.upload_fileobj(file.file, BUCKET_NAME, file.filename)
        return {"message": f"File {file.filename} uploaded to S3 bucket {BUCKET_NAME}"}
    except Exception as e:
        return {"error": str(e)}

@app.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    try:
        
        upload_to_s3_bucket(file)
#         temp_dir = tempfile.gettempdir()

# # Define your custom filename
#         custom_filename = file.filename

#         # Construct the full path
#         temp_file_path = os.path.join(temp_dir, custom_filename)
#         with open(temp_file_path, "wb") as temp_file:
#             temp_file.write(await file.read())
        
#         print(temp_file_path)
#         # Process the PDF using read_pdf_content
        process_file(file.filename)
        
        
        
        # pprint.pprint(documents[0].page_content)
        # os.remove(temp_file_path)
    except Exception as e:
        print(f"Error: {str(e)}")

@app.post('/send-msg')
async def send_msg(history: str = Form(...), message: str = Form(...),file: UploadFile = File(...)):
    print(message)
    try:
        history = eval(history)


        # Construct the full path
        # with open(temp_file_path, "wb") as temp_file:
        #     temp_file.write(await file.read())

        # Process the PDF using read_pdf_content
        # print(temp_file_path, message, history)
        history = chat_with_pdf(file.filename, message, history)
        return {"message": "Processed successfully", "history": history}
    except Exception as e:
        print(f"Error: {str(e)}")

def read_pdf_content(file_name):
    # if file_path is None:
    #     return None 
    
    # loader  = PyMuPDFLoader(file_path)
    # documents = loader.load()
    # return documents
    try:
        # Get the PDF file from S3
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_name)
        pdf_content = response["Body"].read()

        # Extract text using pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Return the text as a Document object
        # document = Document(page_content=text, metadata={"bucket": BUCKET_NAME, "key": file_name})
        # print(document)
        return text

    except Exception as e:
        print("Error reading PDF from S3:", e)
        raise

def chunk_document(documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 80
        )
        chunks = text_splitter.split_text(documents)
        # print(chunks)
        ids = []
        contents = []
        for chunk in chunks:
            id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            contents.append(chunk)
            ids.append(id)
        return contents, ids
    except Exception as e:
        print("Error splitting documents:", e)
        raise

def get_or_create_vectorstore(file_name:str):
    
    collection = client.get_or_create_collection(name= file_name , embedding_function=embedding_functions.DefaultEmbeddingFunction())
    
    return collection


def process_file(file_name):
    if file_name is None:
        return None
    
    documents = read_pdf_content(file_name)
    
    if not documents:
        return None 
    
    chunks,ids = chunk_document(documents)
    # print(chunks)
    collection = get_or_create_vectorstore(file_name)
    # print()
    collection.add(
        documents=chunks, 
        ids = ids)


def chat_with_pdf(file_name, message, history):
    if not message:
        return history
        
    try:
        collection = get_or_create_vectorstore(file_name)
        collection = client.get_collection(name = file_name)
        results = collection.query(query_texts = [message], n_results =3)
        if not results:
            history.append({
                "role": "assistant",
                "content": "Không tìm thấy dữ liệu liên quan trong PDF"
            })
            return history 
        
        # print(results)
        CONTEXT = ""
        for document in results["documents"]:
            CONTEXT += str(document) + "\n\n"
            
        prompt = f"""
            Use the following CONTEXT to answer the QUESTION at the end.
            if you don't know the answer or unsure of the answer, just say that you don't know, don't try to make up an answer.
            Use an unbiased and journalistic tone.
            
            CONTEXT: {CONTEXT}
            QUESTION: {message}
        
        """

        # print(prompt)
        
        response = open_ai_client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        
        history.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
        # print(response.choices[0].message.content)
        return history 
    except Exception as e:
        print('error', e)
        
        return history
    
    
    