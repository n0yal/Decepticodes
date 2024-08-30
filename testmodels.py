import google.generativeai as genai
for model_info in genai.list_tuned_models():
    print(model_info.name)