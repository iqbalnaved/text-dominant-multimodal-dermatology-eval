from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

model_id = "o1-2024-12-17"

# Inspect model metadata
model = client.models.retrieve(model_id)
print(json.dumps(model.model_dump(), indent=2))

# Best practice: set them explicitly yourself
resp = client.responses.create(
    model=model_id,
    input="Say hello.",
)

print(resp)


model_id = "gpt-5.2-2025-12-11"

# Inspect model metadata
model = client.models.retrieve(model_id)
print(json.dumps(model.model_dump(), indent=2))

# Best practice: set them explicitly yourself
resp = client.responses.create(
    model=model_id,
    input="Say hello.",
)

print(resp)