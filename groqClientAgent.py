from python_a2a import BaseA2AClient, Message, MessageRole, TextContent
import requests

class GroqA2AClient(BaseA2AClient):
    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url

    def send_message(self, message: Message) -> Message:
        """
        Sends a single message to the server and returns the response Message.
        Assuming the server accepts POST JSON requests at /a2a endpoint.
        """
        # Prepare payload for server
        payload = {
            "role": message.role.value,
            "content": {
                "text": message.content.text,
                "type": message.content.type,
            }
        }
        response = requests.post(f"{self.server_url}/a2a", json=payload)
        response.raise_for_status()
        data = response.json()

        # Construct response Message from server response
        resp_content = TextContent(text=data["content"]["text"], type=data["content"]["type"])
        resp_role = MessageRole[data["role"].upper()]  # convert string to enum

        return Message(content=resp_content, role=resp_role)

    def send_conversation(self, messages: list[Message]) -> Message:
        """
        Sends a sequence of messages to the server.  
        Here, for simplicity, send only the last message.
        """
        if not messages:
            raise ValueError("No messages to send")

        # Example: send only last message
        return self.send_message(messages[-1])

    def send_prompt(self, prompt: str):
        message = Message(content=TextContent(text=prompt, type="text"), role=MessageRole.USER)
        response = self.send_message(message)
        print("Server Response:", response.content.text)


# if __name__ == "__main__":
#     client = GroqA2AClient(server_url="http://127.0.0.1:5004")
#     user_input = input("Enter prompt: ")
#     client.send_prompt(user_input)

if __name__ == "__main__":
    client = GroqA2AClient(server_url="http://127.0.0.1:5004")
    
    while True:
        user_input = input("Enter prompt: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        client.send_prompt(user_input)
