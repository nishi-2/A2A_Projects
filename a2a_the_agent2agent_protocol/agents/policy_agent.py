import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Add it to a .env file in this directory.")

_global_client = OpenAI()


SYSTEM_PROMPT = """You are an expert insurance agent designed to assist with
coverage queries. Use the provided documents to answer questions
about insurance policies. If the information is not available in
the documents, respond with 'I don't know'"""



class PolicyAgent:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = _global_client
        self.model = model
        with Path("2026AnthemgHIPSBC.pdf").open("rb") as file:
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")
            self.pdf_filename = "2026AnthemgHIPSBC.pdf"


    def answer_query(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": self.pdf_filename,
                            "file_data": f"data:application/pdf;base64,{self.pdf_data}",
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            max_output_tokens=1024,
        )
        return response.output_text


'''
if __name__ == "__main__":
    print("Running Health Insurance Policy Agent")
    agent = PolicyAgent()
    prompt = "How much would I pay for mental health therapy?"
    response = agent.answer_query(prompt)
    print(response)
'''