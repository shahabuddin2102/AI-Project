import base64
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from mcpServerTool import handle_call_tool, handle_read_resource

logger = logging.getLogger("FastAPI-App")
logging.basicConfig(level=logging.INFO)

QA_SUPPORTED_FORMATS = []
SOLUTIONS_SUPPORTED_FORMATS = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global QA_SUPPORTED_FORMATS, SOLUTIONS_SUPPORTED_FORMATS
    logger.info("Starting MCP server...")
    try:
        # Load QA generator formats
        qa_cap = await handle_read_resource("qa-generator:/capabilities")
        QA_SUPPORTED_FORMATS = [ext.lower() for ext in qa_cap["supported_formats"]]
        logger.info(f"QA Supported Formats: {QA_SUPPORTED_FORMATS}")

        # Load Solutions generator formats
        sol_cap = await handle_read_resource("solutions-generator:/capabilities")
        SOLUTIONS_SUPPORTED_FORMATS = [ext.lower() for ext in sol_cap["supported_formats"]]
        logger.info(f"Solutions Supported Formats: {SOLUTIONS_SUPPORTED_FORMATS}")

    except Exception as e:
        logger.error(f"Error loading capabilities: {e}", exc_info=True)
        QA_SUPPORTED_FORMATS = []
        SOLUTIONS_SUPPORTED_FORMATS = []

    yield
    logger.info("Shutting down MCP server...")

app = FastAPI(
    title="Groq Questiona-Answer & Solution Generator API",
    description="Single endpoint for text/file processing",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

from settings import (
    QA_DEFAULT_LIMIT, 
    QA_MIN_LIMIT, 
    QA_MAX_LIMIT, 
    QA_DEFAULT_START,

    SOLUTION_DEFAULT_LIMIT, 
    SOLUTION_MIN_LIMIT, 
    SOLUTION_MAX_LIMIT, 
    SOLUTION_DEFAULT_START
)

@app.post("/generate_qa")
async def generate_qa_via_mcp(
    file: UploadFile = File(...),
    start: int = Query(QA_DEFAULT_START, ge=0),
    limit: int = Query(QA_DEFAULT_LIMIT, ge=QA_MIN_LIMIT, le=QA_MAX_LIMIT)
):
    try:
        file_bytes = await file.read()
        logger.info(f"QA API called with file {file.filename}")

        result = await handle_call_tool("generate_qa", {
            "file_content": base64.b64encode(file_bytes).decode("utf-8"),
            "filename": file.filename,
            "start": start,
            "limit": limit
        })
        return result
    except Exception as e:
        logger.error(f"QA Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_solutions")
async def generate_solution_via_mcp(
    file: UploadFile = File(...),
    chunk_limit: int = Query(SOLUTION_DEFAULT_LIMIT, ge=SOLUTION_MIN_LIMIT, le=SOLUTION_MAX_LIMIT),
    start_chunk: int = Query(SOLUTION_DEFAULT_START, ge=0)
):
    try:
        file_bytes = await file.read()
        filename = file.filename
        logger.info(f"Solution API called with file {filename}")

        result = await handle_call_tool("generate_solution", {
            "content_type": "file",
            "chunk_limit": chunk_limit,
            "start_chunk": start_chunk,
            "file_data": {
                "content": base64.b64encode(file_bytes).decode("utf-8"),
                "filename": filename,
                "mime_type": file.content_type
            }
        })
        return result
    except Exception as e:
        logger.error(f"Solution Endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    print("server is running successfully")
    uvicorn.run(app, host="127.0.0.1", port=8050)

