from typing import List, Dict, Any, Optional
import google.generativeai as genai
from .llm_service import LLMService


class GeminiLLMService(LLMService):
    """
    Google Gemini implementation of LLMService
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def generate_text(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate text using Google Gemini
        """
        # In Python, async calls to external APIs need to be handled carefully
        # For now, we'll use a synchronous call wrapped in a thread to avoid blocking
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def sync_generate():
            if context:
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
            else:
                full_prompt = prompt
                
            response = self.model.generate_content(full_prompt)
            return response.text if response.text else "No response generated"
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, sync_generate)
        
        return result
    
    async def generate_rag_response(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate a RAG-based response using Google Gemini
        """
        # Create a prompt that combines the context and question in a way that Gemini understands
        prompt = f"""
        Based on the following context, answer the question. If the context does not contain enough information to answer the question, respond with a refusal message.

        Context: {context}

        Question: {question}

        Answer:
        """
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def sync_generate():
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    # Check if the response indicates insufficient context
                    response_text = response.text.strip()
                    
                    # If Gemini generates a refusal or indicates no relevant info
                    if any(phrase.lower() in response_text.lower() 
                           for phrase in ["cannot answer", "no relevant", "not mentioned", "not found in context"]):
                        return {
                            "answer": None,
                            "status": "insufficient_context",
                            "confidence_score": None,
                            "reason_code": "NO_RELEVANT_CONTEXT",
                            "explanation": "No relevant context found in the provided information."
                        }
                    
                    return {
                        "answer": response_text,
                        "status": "success",
                        "confidence_score": 0.85  # Default confidence score for Gemini
                    }
                else:
                    return {
                        "answer": None,
                        "status": "insufficient_context", 
                        "confidence_score": None,
                        "reason_code": "NO_RELEVANT_CONTEXT",
                        "explanation": "No response was generated from the model."
                    }
            except Exception as e:
                return {
                    "answer": None,
                    "status": "error",
                    "confidence_score": None,
                    "reason_code": "ERROR",
                    "explanation": f"Error generating response: {str(e)}"
                }
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, sync_generate)
        
        return result


class SelectedTextGeminiLLMService(GeminiLLMService):
    """
    Specialized Gemini service for selected-text-only answering
    Enforces that answers are based only on the provided selected text
    """
    
    async def generate_rag_response(self, question: str, context: str) -> Dict[str, Any]:
        """
        Generate a RAG response that is strictly limited to the selected text context
        """
        prompt = f"""
        You are a helpful assistant that answers questions only based on the provided selected text.
        Do not use any external knowledge or information beyond what is provided in the selected text.
        
        Selected text: {context}
        
        Question: {question}
        
        Answer (based ONLY on the selected text): 
        """
        
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def sync_generate():
            try:
                response = self.model.generate_content(prompt)
                
                if response.text:
                    response_text = response.text.strip()
                    
                    # Check if Gemini indicates it can't answer from the provided text
                    if any(phrase.lower() in response_text.lower() 
                           for phrase in ["cannot answer", "no relevant", "not mentioned", "not found in selected text", "not provided in selected text", "outside the provided text"]):
                        return {
                            "answer": None,
                            "status": "insufficient_context",
                            "confidence_score": None,
                            "reason_code": "NO_RELEVANT_CONTEXT",
                            "explanation": "No relevant information found in the selected text to answer the question."
                        }
                    
                    return {
                        "answer": response_text,
                        "status": "success",
                        "confidence_score": 0.85
                    }
                else:
                    return {
                        "answer": None,
                        "status": "insufficient_context",
                        "confidence_score": None,
                        "reason_code": "NO_RELEVANT_CONTEXT",
                        "explanation": "No response was generated from the model."
                    }
            except Exception as e:
                return {
                    "answer": None,
                    "status": "error",
                    "confidence_score": None,
                    "reason_code": "ERROR",
                    "explanation": f"Error generating response: {str(e)}"
                }
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, sync_generate)
        
        return result