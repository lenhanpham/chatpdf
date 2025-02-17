import os

class Settings:
    def __init__(self):
        self.openrouter_api_key_gemini_fl_2 = os.getenv("OPENROUTER_API_KEY_GEMINI_FL_2")
        self.openrouter_api_key_gemini_pro_2 = os.getenv("OPENROUTER_API_KEY_GEMINI_PRO_2")
        self.openrouter_api_key_ds_r1llama70b = os.getenv("OPENROUTER_API_KEY_DS_R1LLAMA70B")
        self.openrouter_api_key_ds_r1 = os.getenv("OPENROUTER_API_KEY_DS_R1")
        self.google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

settings = Settings()

MODELS = {
    "Google Gemini 2.0 Flash Lite (OpenRouter)": {
        "model_name": "google/gemini-2.0-flash-lite-preview-02-05:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_GEMINI_FL_2"),
        "type": "openrouter",
    },
    "Google Gemini 2.0 Pro (OpenRouter)": {
        "model_name": "google/gemini-2.0-pro-exp-02-05:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_GEMINI_PRO_2"),
        "type": "openrouter",
    },
    "Deepseek R1 Distill Llama 70B (OpenRouter)": {
        "model_name": "deepseek/deepseek-r1-distill-llama-70b:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_DS_R1LLAMA70B"),
        "type": "openrouter",
    },
    "Deepseek R1 (OpenRouter)": {
        "model_name": "deepseek/deepseek-r1:free",
        "api_key": os.getenv("OPENROUTER_API_KEY_DS_R1"),
        "type": "openrouter",
    },
    "Google Gemini 2.0 Flash (Google API)": {
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GOOGLE_GEMINI_API_KEY"),
        "type": "google",
    },
    "Groq Llama 3.3 70B Versatile": {
        "model_name": "llama-3.3-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY"),
        "type": "groq",
    },
}