"""
Helpers for constructing model prompts.

Provides functions to build standardized prompts for QA-pair and coherence models.
"""
from typing import Any, Dict


def build_qa_prompt(question: str, answer: str) -> str:
    """
    Build a composite prompt for QA-pair analysis models.

    Args:
        question (str): The question text.
        answer (str): The answer text.

    Returns:
        str: Formatted prompt combining question and answer.
    """
    return f"Question: {question}\nAnswer: {answer}"


def build_coherence_prompt(context: str, response: str) -> str:
    """
    Build a prompt for coherence analysis between context and response.

    Args:
        context (str): The context or monologue text.
        response (str): The response or answer text.

    Returns:
        str: Formatted prompt for coherence classification.
    """
    return f"Context: {context}\nResponse: {response}"