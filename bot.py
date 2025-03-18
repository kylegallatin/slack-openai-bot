import os
from typing import List

from dotenv import load_dotenv
from slack_bolt import App

from providers import OpenAIProvider, VertexProvider

load_dotenv()

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
INSTRUCTIONS = os.environ.get("INSTRUCTIONS")
AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").lower()

# Initialize the appropriate AI provider
if AI_PROVIDER == "openai":
    ai_provider = OpenAIProvider(api_key=OPENAI_API_KEY)
elif AI_PROVIDER == "vertex":
    ai_provider = VertexProvider(project_id=VERTEX_PROJECT_ID)
else:
    raise ValueError(f"Unsupported AI provider: {AI_PROVIDER}")

app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET
)

@app.event("app_mention")
def handle_mention(event, say):
    text_input = event["text"]
    try:
        response = ai_provider.generate(
            input_text=text_input,
            tools=[{"type": "web_search_preview"}],
            instructions=INSTRUCTIONS
        )
        say(response)
    except Exception as e:
        say(f"An error occurred: {e}")

if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
