# CSVSearchAgent.py
from python_a2a import BaseA2AServer, run_server, Message, TextContent, MessageRole
import pandas as pd
import os

class CSVSearchAgent(BaseA2AServer):
    def __init__(self, csv_path: str):
        super().__init__()
        self.df = pd.read_csv(csv_path)

    def handle_message(self, message: Message) -> Message:
        user_input = message.content.text.lower().strip()
        # Exact match or fuzzy match
        matched_rows = self.df[self.df["question"].str.lower().str.contains(user_input)]

        if not matched_rows.empty:
            # Return first matched answer
            answer = matched_rows.iloc[0]["answer"]
        else:
            answer = "No relevant answer found in CSV."

        return Message(role=MessageRole.AGENT, content=TextContent(text=answer))

if __name__ == "__main__":
    csv_file = "../data/query_result.csv"
    agent = CSVSearchAgent(csv_path=csv_file)
    run_server(agent, host="127.0.0.1", port=5003)
