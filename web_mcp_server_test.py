from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
import os
import httpx
import json
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
LANGSEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"

if not LANGSEARCH_API_KEY:
    raise ValueError("Lansearch api key is missing")

mcp = FastMCP(name="LangSearchTools")

# === Tool Input Schema ===
class LangSearchInput(BaseModel):
    url: str = Field(..., description="The website URL to search from.")
    query: str = Field(..., description="The user question or prompt.")
    langsearch_api_key: str = Field(..., description="Your LangSearch API key.")

@mcp.tool()
async def ask_from_website(query: str) -> dict:
    """Search the web using LangSearch API and return concise result + source link."""
    langsearch_api_key = os.getenv("LANGSEARCH_API_KEY")

    try:
        headers = {
            "Authorization": f"Bearer {langsearch_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": 1
        }

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                "https://api.langsearch.com/v1/web-search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            # print(result, "===============================")

            # Get the first result dict safely
            search_results = result.get("data", {}).get("webPages", {}).get("value", [])

            output = []
            for item in search_results[:3]:
                full_summary = item.get("summary", "No summary found.")
                source_url = item.get("url", "No source URL available.")
                concise_summary = ". ".join(full_summary.split(". ")[:3])

                output.append({
                    "data_source": "web_result",
                    "summary": concise_summary.strip(),
                    "source": source_url
                })
            # print(output, "-----------------------------------")

            return output

    except Exception as e:
        return [{
            "type": "error",
            "summary": f"Error: {str(e)}",
            "source": "N/A"
        }]

# === FastAPI App Mount ===
app = FastAPI(title="LangSearch MCP Server")

async def call_groq_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You're a smart AI agent. If a user asks something related to online data, you must call the 'ask_from_website' tool."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 1,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            print(result, "------------------------------")
            return result["choices"][0]["message"]["content"]
    except httpx.ConnectTimeout:
        return "Error: Groq API timeout. Please try again later."


from fastapi.responses import JSONResponse


@app.post("/mcp/auto-answer")
async def auto_answer(request: Request):
    body = await request.json()
    query = body.get("query", "")

    # Step 1: LLM decides whether to call the tool
    llm_response = await call_groq_llm(f"The user asked: {query}. Should we use ask_from_website tool?")
    print("Groq LLM Response:", llm_response)

    # Step 2: If yes, call tool
    if "ask_from_website" in llm_response.lower():
        result = await mcp.call_tool(name="ask_from_website", arguments={"query": query})
    else:
        result = {
            "type": "text",
            "text": "The LLM decided not to call the tool based on the input query."
        }

    # === Same Serialization Logic as /mcp/tool-call ===
    def serialize(obj):
        if isinstance(obj, TextContent):
            try:
                return json.loads(obj.text)
            except:
                return {"text": obj.text}
        return obj

    return JSONResponse(content=json.loads(json.dumps(result, default=serialize)))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)