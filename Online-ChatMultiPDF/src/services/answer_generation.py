import os
import logging
import requests
import google.generativeai as genai
from groq import Groq
from config.settings import MODELS


# Define the prompt template with better summarization handling
template = """
You are a knowledgeable assistant analyzing PDF documents. Based on the type of question:

For summarization requests:
- Start with a brief overview
- Present key points and findings
- Highlight significant conclusions
- End with a brief synthesis

For specific questions:
- Provide a direct and focused answer
- If the information isn't in the context, clearly state that

Use basic markdown formatting only (**, *, _). Do not use LaTeX or HTML.

Question: {question}
Context: {context}
Answer:
"""

# Function to generate answers using OpenRouter, Google Gemini, or Groq API
MAX_CONTEXT_LENGTH = 4096  # Adjust based on the model's token limit

def generate_answer(question, context, selected_model):
    # Truncate context if it exceeds the maximum length
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH]

    # Format the prompt using the template
    formatted_prompt = template.format(question=question, context=context)

    # Determine which API to use based on the selected model
    model_info = MODELS[selected_model]
    if model_info["type"] == "openrouter":
        # Use OpenRouter API
        data = {
            "model": model_info["model_name"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt},  # Use the formatted prompt here
            ]
        }
        headers = {
            "Authorization": f"Bearer {model_info['api_key']}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(f"{os.getenv('OPENROUTER_BASE_URL')}/chat/completions", headers=headers, json=data)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            # Parse the response JSON
            response_data = response.json()
            if not response_data or "choices" not in response_data or len(response_data["choices"]) == 0:
                logging.error("Invalid API response: Missing 'choices' key or empty list.")
                return None

            # Extract the answer from the response
            first_choice = response_data["choices"][0]
            if "message" not in first_choice or "content" not in first_choice["message"]:
                logging.error("Invalid API response: Missing 'message' or 'content' key.")
                return None

            answer = first_choice["message"]["content"]
            return answer

        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error while generating answer: {e}")
            return None

    elif model_info["type"] == "google":
        # Use Google Gemini API
        try:
            genai.configure(api_key=model_info["api_key"])
            model = genai.GenerativeModel(model_info["model_name"])
            response = model.generate_content(formatted_prompt)  # Use the formatted prompt here
            return response.text
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None

    elif model_info["type"] == "groq":
        # Use Groq API
        try:
            client = Groq(api_key=model_info["api_key"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": formatted_prompt},  # Use the formatted prompt here
                ],
                model=model_info["model_name"],
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            return None





