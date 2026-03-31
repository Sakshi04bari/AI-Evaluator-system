#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


# =========================
# IMPORTS
# =========================

from app import GROQ_API_KEY
import gradio as gr
from PIL import Image

from groq import Groq
import json
import time
import traceback
import os


import pytesseract


from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY")
# API TOKEN
# =========================




client = Groq(api_key=GROQ_API_KEY)

# =========================
# MODEL CONFIG
# =========================
MODEL_NAME = "llama-3.3-70b-versatile"

# =========================
# CLEAN JSON
# =========================
def clean_json(text):
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def safe_parse_json(raw_text):
    cleaned = clean_json(raw_text)

    try:
        return json.loads(cleaned)
    except Exception:
        return {
            "error": "Invalid JSON from model",
            "raw_output": cleaned
        }


# =========================
# NORMALIZE RESULT
# =========================
def normalize_result(result):
    required_keys = [
        "semantic", "relevance", "completeness", "readability",
        "keywords", "consistency", "coherence", "factual",
        "concept", "precision", "recall", "f1",
        "final_score", "feedback"
    ]

    for key in required_keys:
        if key not in result:
            result[key] = 0 if key != "feedback" else "No feedback"

    return result


# =========================
# FINAL SCORE CALCULATION
# =========================
def calculate_final_score(scores):
    weights = {
        "semantic": 0.15,
        "relevance": 0.10,
        "completeness": 0.10,
        "readability": 0.05,
        "keywords": 0.10,
        "consistency": 0.05,
        "coherence": 0.05,
        "factual": 0.10,
        "concept": 0.10,
        "precision": 0.08,
        "recall": 0.07,
        "f1": 0.05
    }

    final_score = 0

    for key, weight in weights.items():
        value = scores.get(key, 0)
        try:
            value = float(value)
        except:
            value = 0

        final_score += value * weight

    return round(final_score, 3)


# =========================
# OCR FUNCTION
# =========================
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"


# =========================
# GROQ CALL
# =========================
def call_groq(prompt):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )

            return response.choices[0].message.content, MODEL_NAME

        except Exception as e:
            err = str(e)
            print(f"Attempt {attempt+1} failed:", err)

            if "429" in err or "rate_limit" in err.lower():
                time.sleep(10)
                continue

            if "500" in err or "503" in err:
                time.sleep(10)
                continue

            return None, MODEL_NAME

    return None, MODEL_NAME


# =========================
# MAIN FUNCTION
# =========================
def evaluate_answer(question, text, image):
    try:
        if not question.strip():
            return {"error": "Question required"}, "Enter question", ""

        if (not text.strip()) and image is None:
            return {"error": "Answer required"}, "Enter answer or image", ""

        extracted_text = ""
        if image:
            extracted_text = extract_text_from_image(image)

        combined_answer = text.strip()

        if extracted_text and not extracted_text.startswith("OCR Error"):
            combined_answer += "\n" + extracted_text

        if not combined_answer.strip():
            return {"error": "Empty answer"}, "No valid text found", extracted_text

        # =========================
        # PROMPT
        # =========================
        prompt = f"""
You are an advanced academic evaluator.

QUESTION:
{question}

STUDENT ANSWER:
{combined_answer}

Evaluate using scores from 0.0 to 1.0:

semantic, relevance, completeness, readability, keywords,
consistency, coherence, factual, concept,
precision, recall, f1

RULES:
- Be strict
- f1 = harmonic mean of precision and recall
- Return ONLY JSON

FORMAT:
{{
 "semantic":0,
 "relevance":0,
 "completeness":0,
 "readability":0,
 "keywords":0,
 "consistency":0,
 "coherence":0,
 "factual":0,
 "concept":0,
 "precision":0,
 "recall":0,
 "f1":0,
 "feedback":""
}}
"""

        raw_output, model_used = call_groq(prompt)

        if raw_output is None:
            return {"error": "Model failed"}, "Model error", extracted_text

        result = safe_parse_json(raw_output)

        if "error" in result:
            return result, "Invalid JSON", extracted_text

        result = normalize_result(result)

        # =========================
        # CALCULATE FINAL SCORE
        # =========================
        result["final_score"] = calculate_final_score(result)
        result["model_used"] = model_used

        return result, result["feedback"], extracted_text

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}, str(e), ""


# =========================
# 🌙 DARK THEME
# =========================
theme = gr.themes.Base(
    primary_hue="cyan",
    neutral_hue="cyan"   # ✅ remove gray completely
).set(
    body_background_fill="#020617",

    # 🌊 Cyan blocks (no gray)
    block_background_fill="#0f172a",
    block_border_color="#06b6d4",

    input_background_fill="#020617",
    input_border_color="#22d3ee",

    # 🌊 Buttons
    color_accent="#06b6d4",
    color_accent_soft="#67e8f9",

    # ✅ Text
    body_text_color="#ffffff",
    block_title_text_color="#22d3ee",
    block_label_text_color="#22d3ee"
)

# =========================
# UI
# =========================
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## 🌙 NeuroGrade AI Evaluator (Dark Mode)")

    question_input = gr.Textbox(label="Question", lines=2)

    with gr.Row():
        text_input = gr.Textbox(label="Answer", lines=8)
        image_input = gr.Image(type="filepath", label="Upload Image")

    btn = gr.Button("Evaluate")

    output = gr.JSON(label="Scores")
    feedback = gr.Textbox(label="Feedback")
    extracted = gr.Textbox(label="Extracted Text")

    btn.click(
        evaluate_answer,
        inputs=[question_input, text_input, image_input],
        outputs=[output, feedback, extracted]
    )
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)


# In[ ]:




