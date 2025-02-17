def select_model(model_name):
    available_models = {
        "Google Gemini 2.0 Flash Lite": {
            "model_name": "google/gemini-2.0-flash-lite-preview-02-05:free",
            "type": "openrouter",
        },
        "Google Gemini 2.0 Pro": {
            "model_name": "google/gemini-2.0-pro-exp-02-05:free",
            "type": "openrouter",
        },
        "Deepseek R1 Distill Llama 70B": {
            "model_name": "deepseek/deepseek-r1-distill-llama-70b:free",
            "type": "openrouter",
        },
        "Deepseek R1": {
            "model_name": "deepseek/deepseek-r1:free",
            "type": "openrouter",
        },
        "Google Gemini 2.0 Flash": {
            "model_name": "gemini-2.0-flash",
            "type": "google",
        },
        "Groq Llama 3.3 70B Versatile": {
            "model_name": "llama-3.3-70b-versatile",
            "type": "groq",
        },
    }

    if model_name in available_models:
        return available_models[model_name]
    else:
        raise ValueError(f"Model '{model_name}' is not available.")

def configure_model(model_info):
    if model_info["type"] == "openrouter":
        return {
            "api_key": model_info.get("api_key"),
            "model_name": model_info["model_name"],
        }
    elif model_info["type"] == "google":
        return {
            "api_key": model_info.get("api_key"),
            "model_name": model_info["model_name"],
        }
    elif model_info["type"] == "groq":
        return {
            "api_key": model_info.get("api_key"),
            "model_name": model_info["model_name"],
        }
    else:
        raise ValueError("Unsupported model type.")