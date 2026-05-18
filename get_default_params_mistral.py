import requests
import os

response = requests.get(
    "https://api.mistral.ai/v1/models",
    headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"}
)

for model in response.json()["data"]:
    if model["id"] == "pixtral-large-2411":
        print(model)
        break