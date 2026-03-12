import os
from dotenv import load_dotenv

from strands.models.openai import OpenAIModel

# load environment variables from .env
load_dotenv()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.abspath(
    os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data", "public_lev_1"))
)
DATA_DIRS = os.getenv("DATA_DIRS", "")
RESULT_FILENAME = os.getenv("RESULT_FILENAME", "result.txt")

# ------------ model configuration ------------
MODEL_ID = "gpt-4o-mini"

# Note: we construct the model lazily in case someone wants to reconfigure
# or perform tests.  `model` and `model_id` are exported for convenience.

def make_model():
    return OpenAIModel(
        client_args={
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
        },
        model_id=MODEL_ID,
        params={
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    )

model = make_model()
