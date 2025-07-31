from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx
import json
import os

load_dotenv()
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
LANGSEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"

if not LANGSEARCH_API_KEY:
    raise ValueError("Lansearch api key is missing")

# === Initialize MCP ===
mcp = FastMCP(name="LangSearchTools")

# === Tool Input Schema ===
class LangSearchInput(BaseModel):
    url: str = Field(..., description="The website URL to search from.")
    query: str = Field(..., description="The user question or prompt.")
    langsearch_api_key: str = Field(..., description="Your LangSearch API key.")

# === MCP Tool Definition ===
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
                    "type": "web_result",
                    "summary": concise_summary.strip(),
                    "source": source_url
                })
            print(output, "-----------------------------------")

            return output

    except Exception as e:
        return [{
            "type": "error",
            "summary": f"Error: {str(e)}",
            "source": "N/A"
        }]

# === FastAPI App Mount ===
app = FastAPI(title="LangSearch MCP Server")

# Custom route works fine now
@app.post("/mcp/tool-call")
async def call_tool(request: Request):
    body = await request.json()
    name = body.get("name")
    arguments = body.get("arguments", {})

    result = await mcp.call_tool(name=name, arguments=arguments)

    def serialize(obj):
        if isinstance(obj, TextContent):
            try:
                return json.loads(obj.text)
            except:
                return {"text": obj.text}
        return obj

    return JSONResponse(content=json.loads(json.dumps(result, default=serialize)))


# === Run the App ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
