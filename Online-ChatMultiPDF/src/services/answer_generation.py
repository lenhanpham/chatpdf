import logging
from .model_handler import ModelHandler

def generate_answer(question, context, selected_model):
    """Generate an answer using the selected model"""
    try:
        model_handler = ModelHandler()
        return model_handler.generate_response(question, context, selected_model)
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return None
