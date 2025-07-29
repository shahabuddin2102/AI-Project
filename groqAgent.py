import os
import requests
from dotenv import load_dotenv
from python_a2a import BaseA2AServer, run_server
from python_a2a import Message, MessageRole, TextContent

load_dotenv()

class GroqA2AServer(BaseA2AServer):
    def __init__(self, api_key: str, model: str, system_prompt: str):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def _call_llm(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1024,
            }
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling Groq LLM API: {e}")
            return f"Error: {e}"


    def handle_message(self, message: Message) -> Message:
        try:
            print(f"[A2A] Incoming message: {message}")

            prompt = message.content.text if isinstance(message.content, TextContent) else ""

            response_text = self._call_llm(prompt)
            print(f"[A2A] LLM response: {response_text}")

            # Create a TextContent object for the response
            response_content = TextContent(text=response_text, type="text")

            # Create and return a new Message with role AGENT and the response content
            return Message(content=response_content, role=MessageRole.AGENT)

        except Exception as e:
            print(f"[A2A] Error in handle_message: {e}")

            error_content = TextContent(text=f"Error: {e}", type="text")
            return Message(content=error_content, role=MessageRole.SYSTEM)


if __name__ == "__main__":
    agent = GroqA2AServer(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192",
        system_prompt="You are a helpful AI assistant."
    )


    print("Testing Groq LLM API with prompt: 'Hello'")
    test_response = agent._call_llm("Hello")
    print("Test response:", test_response)

    # Run the server
    run_server(agent, host="127.0.0.1", port=5004)
