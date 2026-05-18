import google.generativeai as genai

genai.configure(api_key="AIzaSyAbE7CCYlf2RmmmEZlYQT_XSYe9ug-X_ck")

model = genai.GenerativeModel("gemini-3-pro-preview")

resp = model.generate_content("Say hi.")
print("GOT:", resp.text)
