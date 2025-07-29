from python_a2a import AgentNetwork, Message, TextContent, MessageRole
from dotenv import load_dotenv
import logging
import os
from openai import OpenAI

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === GROQ setup ===
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def query_groq(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reformulates user questions to match a CSV file's questions."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return prompt  # fallback

class UserClientAgent:
    def __init__(self):
        self.network = AgentNetwork(name="UserClientNetwork")
        self.network.add("csv_agent", "http://127.0.0.1:5003/a2a")

    def ask_csv(self, user_prompt: str) -> str:
        logger.info(f"Original user input: {user_prompt}")

        refined_prompt = query_groq(user_prompt)
        logger.info(f"Refined prompt from Groq: {refined_prompt}")

        message = Message(role=MessageRole.USER, content=TextContent(text=refined_prompt))
        print(message, "-"*50)

        responses = self.network.send("csv_agent", [message], run=True)
        response = responses[0]


        logger.info(f"Response from CSV agent: {response.content.text}")
        return response.content.text



if __name__ == "__main__":
    agent = UserClientAgent()
    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        try:
            response = agent.ask_csv(user_input)
            print(f"Answer: {response}")
        except Exception as e:
            logger.error(f"Error: {e}")
