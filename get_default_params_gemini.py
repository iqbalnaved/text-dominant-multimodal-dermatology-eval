import google.generativeai as genai
import os 

genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model_info = genai.get_model("models/gemini-3-pro-preview")
print(model_info)
