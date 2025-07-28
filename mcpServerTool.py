# ===== MCP Server Tool Implementation Here =====
import base64
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
from typing import List, Dict, Any, Optional

# ===== All Function Imported Below From generateQA & generateSolution file =====
from generateQA import build_prompt, chunk_text, generate_qa_from_text, parse_and_merge_qa_responses
from generateSolution import process_text_content, process_chunks
from utilsFile import extract_text_from_file

# === mcp server setup ===
mcp_server = Server("generator-server")

@mcp_server.list_resources()
async def handle_list_resources() -> List[Resource]:
    return [
        Resource(
            uri="qa-generator:/capabilities",
            name="QA Generator Capabilities",
            description="Information about supported file formats and features",
            mimeType="application/json"
        ),
        Resource(
            uri="solutions-generator:/capabilities",
            name="Solutions Generator Capabilities",
            description="Information about supported file formats and features",
            mimeType="application/json"
        )
    ]

@mcp_server.read_resource()
async def handle_read_resource(uri: str) -> Any:
    if uri == "qa-generator:/capabilities":
        return {
            "supported_formats": ["PDF", "DOCX", "TXT", "CSV"],
            "max_file_size": "100MB",
            "features": ["qa_generation"]
        }
    
    elif uri == "solutions-generator:/capabilities":
        return {
            "supported_formats": ["PDF", "DOCX", "TXT", "CSV"],
            "max_file_size": "100MB",
            "features": ["solutions_generation"]
        }
    raise ValueError(f"Unknown resource: {uri}")

@mcp_server.list_tools()
async def handle_list_tools() -> List[Tool]:
    return [
        Tool(
            name="generate_qa",
            description="Generate Q&A pairs from uploaded files (PDF, DOCX, TXT, CSV)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_content": {"type": "string", "format": "base64"},
                    "filename": {"type": "string"},
                    "start": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["file_content", "filename"]
            }
        ),
        Tool(
            name="generate_solution",
            description="Process uploaded file content only (PDF, DOCX, TXT, CSV)",
            inputSchema={
                "type": "object",
                "properties": {
                    "content_type": {"type": "string", "enum": ["file"]},
                    "file_data": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "filename": {"type": "string"},
                            "mime_type": {"type": "string"}
                        },
                        "required": ["content", "filename", "mime_type"]
                    },
                    "chunk_limit": {"type": "integer"},
                    "start_chunk": {"type": "integer"}
                },
                "required": ["content_type", "file_data"]
            }
        )
    ]

@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> dict:
    if name == "generate_qa":
        file_bytes = base64.b64decode(arguments["file_content"])
        filename = arguments["filename"]
        ext = filename.split(".")[-1].lower()

        full_text = extract_text_from_file(file_bytes, ext)
        chunks = chunk_text(full_text)
        responses = [await generate_qa_from_text(build_prompt(c)) for c in chunks]
        return parse_and_merge_qa_responses(responses)

    elif name == "generate_solution":
        file_data = arguments["file_data"]
        file_bytes = base64.b64decode(file_data["content"])
        ext = file_data["filename"].split(".")[-1].lower()

        documents = process_text_content(extract_text_from_file(file_bytes, ext))
        start = arguments.get("start_chunk", 0)
        limit = arguments.get("chunk_limit", 10)
        results = process_chunks(documents[start:start+limit], limit)

        return results

    else:
        raise ValueError(f"Unknown tool name: {name}")
    
    
