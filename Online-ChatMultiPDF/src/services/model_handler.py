import logging
import os
import requests
import google.generativeai as genai
from groq import Groq
from config.settings import MODELS

class ModelHandler:
    def __init__(self):
        self.template = """
    You are a knowledgeable assistant analyzing PDF documents. Please answer the questions as detailed as possible depending on types of questions.
    
    For summarization requests:
    - Start with the file name.
    - Provide a brief overview of the file's contents.
    - Present key points and findings.
    - Highlight significant conclusions.
    - End with a brief synthesis.
    
    For specific questions:
    - Provide a direct and focused answer.
    - If the information isn't in the context, clearly state that.
    
    Use basic markdown formatting only (**, *, _). Do not use LaTeX or HTML.
    
    Context: {context}
    Question: {question}
    Answer:
        """
        self.max_context_length = 4096

    def format_prompt(self, question, context):
        """Format the prompt with question and truncated context"""
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length]
        return self.template.format(question=question, context=context)

    def handle_openrouter_request(self, model_info, formatted_prompt):
        """Handle requests to OpenRouter API"""
        data = {
            "model": model_info["model_name"],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt},
            ]
        }
        headers = {
            "Authorization": f"Bearer {model_info['api_key']}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                f"{os.getenv('OPENROUTER_BASE_URL')}/chat/completions", 
                headers=headers, 
                json=data
            )
            response.raise_for_status()
            response_data = response.json()
            
            if not response_data or "choices" not in response_data:
                raise ValueError("Invalid API response: Missing 'choices' key")
                
            first_choice = response_data["choices"][0]
            if "message" not in first_choice or "content" not in first_choice["message"]:
                raise ValueError("Invalid API response: Missing message content")
                
            return first_choice["message"]["content"]
        except Exception as e:
            logging.error(f"OpenRouter API error: {e}")
            return None

    def handle_google_request(self, model_info, formatted_prompt):
        """Handle requests to Google Gemini API"""
        try:
            genai.configure(api_key=model_info["api_key"])
            model = genai.GenerativeModel(model_info["model_name"])
            response = model.generate_content(formatted_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None

    def handle_groq_request(self, model_info, formatted_prompt):
        """Handle requests to Groq API"""
        try:
            client = Groq(api_key=model_info["api_key"])
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": formatted_prompt},
                ],
                model=model_info["model_name"],
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            return None

    def generate_response(self, question, context, selected_model):
        """Generate response using the selected model"""
        formatted_prompt = self.format_prompt(question, context)
        model_info = MODELS[selected_model]
        
        handlers = {
            "openrouter": self.handle_openrouter_request,
            "google": self.handle_google_request,
            "groq": self.handle_groq_request
        }
        
        handler = handlers.get(model_info["type"])
        if not handler:
            logging.error(f"Unsupported model type: {model_info['type']}")
            return None
            
        return handler(model_info, formatted_prompt)
