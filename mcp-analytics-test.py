# ===== importing module =====
import os
import io
import re
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from string import Template
from sqlalchemy import create_engine
from fastapi import UploadFile, File
from mcp.server.fastmcp import FastMCP
from contextlib import asynccontextmanager
from requests.exceptions import RequestException

# === Configuration ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_NEW_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
DATABASE_URL = os.getenv("DATABASE_URL")

# PROMPT_TEMPLATE_FILE = "mcp_dashboard_prompt.txt"

# === Logging Configuration ===
log_file = "mcp_analytics_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FastAPI + MCP Setup ===
mcp = FastMCP(name="DataAnalysisServer", stateless_http=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("starting mcp server on /mcp")
    yield
    # Optional: Add shutdown logic here if needed

app = FastAPI(lifespan=lifespan)

class ToolCallResult(BaseModel):
    output: Dict[str, Any]

# === Data Base Connections ===
# def get_db_engine():
#     try:
#         logging.info("Creating DB engine")
#         engine = create_engine(DATABASE_URL)
#         logging.info("DB engine created successfully")
#         return engine
#     except Exception as e:
#         logging.error(f"DB connection error: {e}")
#         raise HTTPException(status_code=500, detail=f"DB connection error: {e}")

# === helper functions ===
def clean_nan(obj):
    if isinstance(obj, (float, np.floating)) and (np.isnan(obj) or pd.isna(obj)):
        return 0
    if obj is None or obj == "NaN":
        return None
    if isinstance(obj, dict):
        cleaned = {k: clean_nan(v) for k, v in obj.items()}
        return {k: v for k, v in cleaned.items() if v is not None and v != {} and v != []}
    if isinstance(obj, list):
        return [clean_nan(x) for x in obj if x is not None and x != {} and x != []]
    return obj


# === KPI Computation ===
def calculate_fallback_kpis(df):
    df = df.fillna({
        "wrapup_time": 0,
        "wait_time": 0,
        "call_duration": 0,
        "call_status_disposition": "unknown",
        "czdisconnectedby": "unknown",
        "agent_name": "unknown",
        "agent_id": 0,
        "location_name": "unknown",
        "campaign_name": "unknown",
        "hold_time": 0,
    })

    # Convert numeric columns
    for col in ["wrapup_time", "wait_time", "call_duration", "hold_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    kpi = {
        "call_duration_avg": round(df["call_duration"].mean(), 2),
        "call_duration_total": int(df["call_duration"].sum()),
        "call_duration_max": int(df["call_duration"].max()),
        "call_duration_min": int(df["call_duration"].min()),
        "top_longest_calls": df.nlargest(1, "call_duration")[["agent_name", "call_duration", "call_status_disposition"]].to_dict(orient="records"),
        "wrapup_time_avg": round(df["wrapup_time"].mean(), 2),
        "wrapup_time_total": int(df["wrapup_time"].sum()),
        "wrapup_time_max": int(df["wrapup_time"].max()),
        "wrapup_time_min": int(df["wrapup_time"].min()),
        "wait_time_avg": round(df["wait_time"].mean(), 2),
        "wait_time_total": int(df["wait_time"].sum()),
        "wait_time_max": int(df["wait_time"].max()),
        "wait_time_min": int(df["wait_time"].min()),
        "call_status_disposition": df["call_status_disposition"].value_counts().to_dict(),
        "disconnected_by": df["czdisconnectedby"].value_counts().to_dict(),
    }

    try:
        top_agent_name = df["agent_name"].value_counts().idxmax()
        top_agent_id = int(df[df["agent_name"] == top_agent_name]["agent_id"].iloc[0])
        kpi["top_agent"] = {"name": top_agent_name, "id": top_agent_id}
    except:
        kpi["top_agent"] = {"name": "unknown", "id": 0}

    kpi["agent_wise_total_calls"] = df["agent_name"].value_counts().to_dict()
    kpi["location_name"] = df["location_name"].mode().iloc[0] if not df["location_name"].isnull().all() else "unknown"
    # kpi["campaign_name"] = df["campaign_name"].mode().iloc[0] if not df["campaign_name"].isnull().all() else "unknown"
    kpi["campaign_name_split"] = df["campaign_name"].value_counts().to_dict()
    kpi["inbound_outbound_split"] = df["campaign_type"].value_counts().to_dict()

    df["call_start_date_time"] = pd.to_datetime(df["call_start_date_time"], errors="coerce")
    df["call_hour"] = df["call_start_date_time"].dt.hour
    kpi["peak_call_hour"] = int(df["call_hour"].value_counts().idxmax())
    kpi["call_volume_by_hour"] = df["call_hour"].value_counts().sort_index().to_dict()

    kpi["unique_customers"] = int(df["cust_ph_no"].nunique())

    kpi["calls_per_channel"] = df["campaign_channel"].value_counts().to_dict()

    answered_calls = df[df["call_status_disposition"] == "answered"]
    kpi["avg_speed_of_answer"] = round(answered_calls["wait_time"].mean(), 2) if not answered_calls.empty else 0
    threshold_seconds = 20
    within_threshold = answered_calls[answered_calls["wait_time"] <= threshold_seconds]
    kpi["service_level"] = round((len(within_threshold) / len(answered_calls)) * 100, 2) if len(answered_calls) > 0 else 0

    kpi["call_transfer_rate"] = round((df["czdisconnectedby"] == "Transfer_Call").mean() * 100, 2)
    kpi["call_outcome_split"] = df["call_status_disposition"].value_counts().to_dict()
    kpi["avg_queue_abandon_time"] = round(df[df["czdisconnectedby"] == "CUSTOMER"]["wait_time"].mean(), 2)

    kpi["hold_return_rate"] = round((df["hold_time"] > 0).mean() * 100, 2)

    try:
        fcr_total = df["cust_ph_no"].nunique()
        repeat_customers = 0
        for cust, group in df.groupby("cust_ph_no"):
            sorted_times = group["call_start_date_time"].sort_values()
            if len(sorted_times) > 1:
                diffs = sorted_times.diff().dt.total_seconds().fillna(999999)
                if any(diffs < 86400):
                    repeat_customers += 1
        kpi["first_call_resolution_percent"] = round(((fcr_total - repeat_customers) / fcr_total) * 100, 2) if fcr_total > 0 else None
    except:
        kpi["first_call_resolution_percent"] = None

    kpi["callback_requests"] = int((pd.to_numeric(df["callback_flag"], errors="coerce").fillna(0) == 1).sum())

    try:
        df["ivr_start_time"] = pd.to_datetime(df["ivr_start_time"], errors="coerce")
        df["ivr_end_time"] = pd.to_datetime(df["ivr_end_time"], errors="coerce")
        total_ivr = df["ivr_start_time"].notna().sum()
        drop_calls = df[(df["ivr_start_time"].notna()) & (df["callpicked"] == 0)].shape[0]
        complete_calls = df[(df["ivr_start_time"].notna()) & (df["callpicked"] == 1)].shape[0]
        kpi["ivr_dropoff_rate"] = round((drop_calls / total_ivr) * 100, 2) if total_ivr > 0 else 0
        kpi["ivr_completion_rate"] = round((complete_calls / total_ivr) * 100, 2) if total_ivr > 0 else 0
    except:
        kpi["ivr_dropoff_rate"] = 0
        kpi["ivr_completion_rate"] = 0

    kpi["after_call_work_time"] = int(df["wrapup_time"].sum())

    # AHT = call duration + wrapup time
    df["aht"] = df["call_duration"] + df["wrapup_time"]
    kpi["average_handling_time"] = round(df["aht"].mean(), 2)

    # Transfer count by agent
    kpi["transfer_count_by_agent"] = df[df["czdisconnectedby"] == "Transfer_Call"]["agent_name"].value_counts().to_dict()

    # Call type split
    kpi["call_type_split"] = df["call_type"].value_counts().to_dict()

    # Agent effectiveness = answered / total per agent
    agent_total = df["agent_name"].value_counts()
    agent_answered = df[df["call_status_disposition"] == "answered"]["agent_name"].value_counts()
    kpi["agent_effectiveness"] = (agent_answered / agent_total).fillna(0).round(2).to_dict()  

    return kpi

# === Prompt Builder ===
def build_groq_prompt(sample_data: list, description: dict, kpi_fallback: dict) -> str:
    prompt_template = Template("""
You are a professional call center data analyst.

Your job is to extract and compute **industry-standard KPIs** from call center data.

You MUST return the **full JSON structure** defined below, matching the exact field names and nested structure.

📌 Rules to follow:
- Use fallback values wherever provided.
- NEVER fabricate numbers.
- If a KPI is not computable, use `null`.
- Do not skip any KPI key — all keys must be present, even if the value is `null`.
- If fallback values are 0 or missing, prefer `null` unless you can confidently compute from sample data.
- All fields must be present as defined below, including advanced insights.

🔒 Output Format:
- Only return a **strictly valid JSON** object.
- Do NOT wrap with markdown (no ```json).
- Do NOT include any explanations or comments.
- Output must start and end with `{}`.
- JSON must be `json.loads()` parsable.

---

📊 **Call Traffic & Volume**
- inbound_outbound_split
- calls_per_channel
- repeat_call_rate
- callback_requests
- ivr_dropoff_rate
- ivr_completion_rate

⏱ **Call Handling**
- avg_speed_of_answer
- service_level

👨‍💼 **Agent Performance**
- agent_wise_total_calls
- agent_occupancy_percent
- agent_utilization_percent
- total_login_time
- total_break_time
- after_call_work_time
- avg_wrapup_time
- agent_availability_percent

🎧 **Disposition & CX**
- first_call_resolution_percent
- call_outcome_split
- escalation_rate
- csat_score
- nps_score

🔁 **Queue Insights**
- avg_queue_abandon_time
- queue_overflow_rate
- call_transfer_rate
- avg_call_waiting_time
- hold_return_rate

💡 **Additional Advanced Insights**
- agent_effectiveness
- call_abandonment_breakdown
- peak_call_load_insights
- customer_behavior
- team_or_skill_performance
- critical_alerts

=== Fallback KPIs ===
$kpi_fallback

=== Sample Records (JSON format) ===
$sample_data

=== Statistical Summary (describe()) ===
$description
""")

    return prompt_template.substitute(
        kpi_fallback=json.dumps(kpi_fallback, indent=2),
        sample_data=json.dumps(sample_data, indent=2),
        description=json.dumps(description, indent=2)
    )


# === Groq API Call ===
def groq_api_call(prompt: str) -> dict:
    logger.info("Calling Groq API for analysis")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict JSON generator for call center KPIs.\n"
                    "Always return the complete JSON structure below.\n"
                    "Use provided fallback values if available.\n"
                    "Never fabricate data.\n"
                    "If any value is missing, use null. Never omit fields.\n"
                    "Only return JSON. No text, no explanation, no markdown."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload)
            logger.info(f"Groq API response status code: {response.status_code}")

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning(f"429 Rate limit hit on attempt {attempt + 1}. Retrying in {retry_delay * (attempt + 1)} seconds...")
                time.sleep(retry_delay * (attempt + 1))
                continue

            response.raise_for_status()

            content = response.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"Raw Groq response:\n{content}")

            # Clean possible markdown
            content_clean = re.sub(r"^```(?:json)?\s*|```$", "", content, flags=re.IGNORECASE).strip()
            logger.info(f"Cleaned content before parsing:\n{content_clean}")

            try:
                return json.loads(content_clean)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON decode failed: {e}")
                match = re.search(r'\{.*\}', content_clean, re.DOTALL)
                if match:
                    cleaned_json = match.group(0)
                    cleaned_json = re.sub(r',\s*([\]}])', r'\1', cleaned_json)
                    cleaned_json = re.sub(r':[ \t]*[^\d{"][^,}\]]*', ':""', cleaned_json)
                    return json.loads(cleaned_json)
                else:
                    logger.error("No JSON object found in Groq response.")
                    logger.error(f"Groq raw content:\n{content}")
                    raise ValueError("Groq returned no parsable JSON object.")

        except RequestException as e:
            logger.error(f"Groq API request failed: {str(e)}")
            raise RuntimeError(f"Groq API request failed: {str(e)}")

    # If all retries fail
    raise RuntimeError("Groq API request failed after multiple retries due to rate limits.")


@mcp.tool("analyze_csv_data")
def analyze_csv_data(records: list[dict]) -> ToolCallResult:
    logger.info("MCP tool invoked: analyze_csv_data")

    try:
        df = pd.DataFrame(records)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")

        sample_data = df.head(10).to_dict(orient="records")
        description = df.describe(include='all').fillna("").to_dict()
        kpi_fallback = calculate_fallback_kpis(df)

        prompt = build_groq_prompt(
            sample_data=sample_data,
            description=description,
            kpi_fallback=kpi_fallback
        )

        logger.info(f"Prompt ready for Groq model")
        try:
            groq_output = groq_api_call(prompt)
            insights = groq_output
            logger.info(f"groq output generated successfully: {insights}")
        except ValueError:
            logger.error("Groq returned invalid JSON. Using fallback KPIs.")
            insights = kpi_fallback

        logger.info("Groq-based analysis completed successfully")

        # Clean NaNs before returning
        safe_data = clean_nan(insights)
        logger.info(f"Final cleaned insights: {safe_data}")
        return ToolCallResult(output=safe_data)

    except Exception as e:
        logger.error(f"Exception during Groq analysis: {str(e)}")
        return ToolCallResult(output={"error": str(e)})


@app.post("/analyze-data")
async def analyze_data(request: Request, file: UploadFile = File(...)):
    logger.info("POST request received at /analyze-data")

    try:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        records = df.tail(100).to_dict(orient="records")

        # MCP call result
        status, result = await mcp.call_tool("analyze_csv_data", {"records": records})
        output = result.get("output", {})

        if not output:
            raise HTTPException(status_code=500, detail="Tool returned no output.")

        logger.info(f"Dynamically called MCP tool successfully. Output: {output}")
        return output

    except Exception as e:
        logger.error(f"Exception during analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount MCP after FastAPI instance is created
app.mount("/tools", mcp.streamable_http_app(), name="mcp")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8020)


