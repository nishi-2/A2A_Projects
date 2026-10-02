import base64
import os
import sys
from pathlib import Path

_agent_dir = Path(__file__).resolve().parent
for _path in (_agent_dir, _agent_dir.parent):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from openai import OpenAI

from common.config import load_env

load_env()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set. Add it to the project .env file.")

_global_client = OpenAI()


SYSTEM_PROMPT = """You are an expert insurance agent designed to assist with
coverage queries. Use the provided documents to answer questions
about insurance policies. If the information is not available in
the documents, respond with 'I don't know'"""



class PolicyAgent:
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = _global_client
        self.model = model
        pdf_path = Path(__file__).resolve().parent / "2026AnthemgHIPSBC.pdf"
        with pdf_path.open("rb") as file:
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