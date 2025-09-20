"""
Legal Translator app with RAG->
FastAPI backend with Google Gemini and RAG pipeline using ChromaDB and LangChain
"""
from dotenv import load_dotenv
import os
import hashlib
from typing import List
from pathlib import Path
import shutil
from datetime import datetime
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Document processing
import pypdf
from docx import Document as DocxDocument

# --- LangChain Core Imports ---
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- LangChain Google Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- ChromaDB Client ---
import chromadb
from chromadb.config import Settings as ChromaSettings

load_dotenv()

# --- Environment and App Setup ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

app = FastAPI(title="Legal Translator AI", version="2.1.0-fixed")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Directory and DB Initialization ---
UPLOAD_DIR = Path("uploads")
CHROMA_DIR = Path("chroma_db")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=ChromaSettings(anonymized_telemetry=False)
)

# --- LangChain AI Model Configuration ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_output_tokens=2048,
)

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
)

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    filename: str
    question: str

class ScanRequest(BaseModel):
    filename: str

class UploadResponse(BaseModel):
    success: bool
    filename: str
    message: str
    chunks_created: int

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

class RiskClause(BaseModel):
    clause_type: str = Field(description="The type of legal clause, e.g., 'Termination', 'Limitation of Liability'.")
    text: str = Field(description="The exact text of the clause from the document (max 200 chars).")
    explanation: str = Field(description="A plain English explanation of what the clause means and its risks.")
    severity: str = Field(description="A risk severity rating: 'high', 'medium', or 'low'.")

class ScanResponse(BaseModel):
    risks: List[RiskClause]
    total_risks: int

# --- Utility Functions ---
def get_collection_name(filename: str) -> str:
    """Creates a consistent, valid collection name from a filename."""
    return hashlib.md5(filename.encode()).hexdigest()

def extract_text(file_path: Path) -> str:
    """Extracts text from various document types."""
    extension = file_path.suffix.lower()
    text = ""
    try:
        if extension == '.pdf':
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        elif extension in ['.docx', '.doc']:
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif extension == '.txt':
            text = file_path.read_text(encoding='utf-8')
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {extension}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading {file_path.name}: {e}")
    return text

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Legal Translator AI API (LangChain Edition) is running"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_path = UPLOAD_DIR / file.filename
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    try:
        # 1. Extract text
        text = extract_text(file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty or unreadable.")

        # 2. Split text into chunks (list of strings)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text chunks from the document.")
        
        # 3. Create a metadata dictionary for each chunk
        metadatas = [{"filename": file.filename}] * len(chunks)

        # 4. Create or replace the ChromaDB collection
        collection_name = get_collection_name(file.filename)
        try:
            chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass  # Collection didn't exist, which is fine

        # 5. Use Chroma.from_texts to create embeddings and add to the DB
        # This is the key fix: It uses raw text and metadata, avoiding the
        # problematic LangChain Document object serialization.
        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings_model,
            metadatas=metadatas,
            client=chroma_client,
            collection_name=collection_name,
        )

        return UploadResponse(
            success=True,
            filename=file.filename,
            message=f"'{file.filename}' processed. Created {len(chunks)} chunks.",
            chunks_created=len(chunks),
        )
    except Exception as e:
        # Log the full traceback for better debugging
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    collection_name = get_collection_name(request.filename)
    try:
        # 1. Load the existing vector store
        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings_model,
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    except Exception:
        # A more robust check to see if the collection actually exists
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if collection_name not in existing_collections:
             raise HTTPException(status_code=404, detail=f"Index for '{request.filename}' not found. Please upload it.")
        else:
            raise HTTPException(status_code=500, detail="Error loading the index. It may be corrupted.")

    # 2. Define the RAG prompt template
    template = """
    Answer the question based ONLY on the following context.
    Explain the answer in simple, plain English, as if talking to a non-lawyer.
    If the context does not contain the answer, state that you cannot find the answer in the document.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Define a function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 4. Build the RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        # First, retrieve documents to get sources separately.
        # Note: This runs the retriever, and the chain will run it again. For this app,
        # the performance impact is negligible, but for production it could be optimized.
        source_docs = retriever.invoke(request.question)
        sources = [doc.page_content[:150] + "..." for doc in source_docs]

        # 5. Invoke the chain to get the answer
        answer = rag_chain.invoke(request.question)
        
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# --- Add the new Pydantic model with the others ---

class RiskAnalysis(BaseModel):
    """A list of identified risk clauses from the document."""
    risks: List[RiskClause]

# --- Then, update the /scan endpoint like this ---





@app.post("/scan", response_model=ScanResponse)
async def scan_for_risks(request: ScanRequest):
    file_path = UPLOAD_DIR / request.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{request.filename}' not found.")
    
    try:
        text = extract_text(file_path)
        
        # Split text into manageable chunks if it's too long
        max_chunk_size = 10000
        text_chunks = []
        
        if len(text) > max_chunk_size:
            # Split into overlapping chunks to avoid missing clauses that span boundaries
            for i in range(0, len(text), max_chunk_size - 500):  # 500 char overlap
                chunk = text[i:i + max_chunk_size]
                text_chunks.append(chunk)
        else:
            text_chunks = [text]
        
        all_risks = []
        
        # Process each chunk
        for chunk in text_chunks:
            # Use a more direct prompt approach instead of structured output
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert legal analyst. Analyze the document text and identify potentially risky legal clauses.

For each risky clause you find, provide:
1. clause_type: The type (e.g., "Indemnification", "Limitation of Liability", "Termination", "Arbitration", "Non-Compete", "Force Majeure")
2. text: The exact clause text (first 200 characters only)
3. explanation: Plain English explanation of the risk
4. severity: "high", "medium", or "low"

Format your response as a JSON array of objects. Example:
[
  {
    "clause_type": "Indemnification",
    "text": "Party A shall indemnify and hold harmless Party B from any claims...",
    "explanation": "This clause makes you responsible for covering legal costs and damages, which could be expensive.",
    "severity": "high"
  }
]

If no risky clauses are found, return an empty array: []"""),
                ("human", "Analyze this document text for risky legal clauses:\n\n{document_text}")
            ])
            
            chain = prompt | llm | StrOutputParser()
            
            try:
                # Get the response from the LLM
                response = chain.invoke({"document_text": chunk})
                
                # Clean up the response to ensure it's valid JSON
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.endswith("```"):
                    response = response[:-3]
                response = response.strip()
                
                # Parse JSON response
                import json
                try:
                    chunk_risks = json.loads(response)
                    
                    # Validate and convert to RiskClause objects
                    for risk_data in chunk_risks:
                        try:
                            # Ensure all required fields are present with defaults
                            risk_clause = RiskClause(
                                clause_type=risk_data.get("clause_type", "Unknown"),
                                text=risk_data.get("text", "")[:200],  # Truncate to 200 chars
                                explanation=risk_data.get("explanation", "No explanation provided"),
                                severity=risk_data.get("severity", "medium").lower()
                            )
                            
                            # Validate severity
                            if risk_clause.severity not in ["high", "medium", "low"]:
                                risk_clause.severity = "medium"
                                
                            all_risks.append(risk_clause)
                        except Exception as e:
                            print(f"Error parsing individual risk: {e}")
                            continue
                            
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Raw response: {response}")
                    # Try to extract information using a fallback method
                    fallback_risks = parse_fallback_response(response)
                    all_risks.extend(fallback_risks)
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        # Remove duplicates based on clause_type and first 50 chars of text
        unique_risks = []
        seen = set()
        
        for risk in all_risks:
            identifier = f"{risk.clause_type}_{risk.text[:50]}"
            if identifier not in seen:
                seen.add(identifier)
                unique_risks.append(risk)
        
        return ScanResponse(risks=unique_risks, total_risks=len(unique_risks))
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error scanning document: {str(e)}")

def parse_fallback_response(response: str) -> List[RiskClause]:
    """Fallback parser when JSON parsing fails"""
    risks = []
    
    # Common risk clause patterns
    risk_patterns = [
        "indemnif", "liable", "liability", "terminate", "termination", 
        "arbitrat", "non-compete", "force majeure", "breach", "default",
        "liquidated damages", "penalty", "confidential", "non-disclosure"
    ]
    
    lines = response.split('\n')
    for line in lines:
        line_lower = line.lower()
        for pattern in risk_patterns:
            if pattern in line_lower and len(line.strip()) > 10:
                risk = RiskClause(
                    clause_type="Identified Risk",
                    text=line.strip()[:200],
                    explanation="Potential legal risk identified in document",
                    severity="medium"
                )
                risks.append(risk)
                break  # Only add once per line
    
    return risks[:5]  # Limit fallback results


@app.get("/documents")
async def list_documents():
    return {
        "documents": [
            {"filename": f.name, "size": f.stat().st_size, "uploaded_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
            for f in UPLOAD_DIR.glob("*") if f.is_file()
        ]
    }

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()
        
    collection_name = get_collection_name(filename)
    try:
        chroma_client.delete_collection(name=collection_name)
    except ValueError:
        pass
        
    return {"message": f"Document '{filename}' and its index deleted."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)