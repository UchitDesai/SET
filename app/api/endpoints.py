from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil
import json

from app.services.memorymanager import memory_manager
from app.services.vector_store import VectorStore
from app.services.language_detector import detect_language
from app.services.phishing_extractor import extract_text_from_file
from app.services.reasoning_graph import build_reasoning_graph
from app.services.cross_modal_reasoner import apply_cross_modal_reasoning
from app.services.adaptive_fusion_engine import fuse_scores
from app.services.llm_interface import generate_explanation
from app.services.modality_processor import (
    process_modalities,
    merge_features,
    compute_modality_context
)

router = APIRouter()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@router.post("/start_session")
def start_session():
    session_id = memory_manager.create_session()
    return {"session_id": session_id}


@router.post("/analyze_phishing")
async def analyze_phishing(
    session_id: str = Form(...),
    raw_text: str = Form(None),
    url: str = Form(None),
    email_subject: str = Form(None),
    email_sender: str = Form(None),
    file: UploadFile = File(None),
):
    if not memory_manager.session_exists(session_id):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid or expired session_id. Call /start_session first."}
        )

    try:
        file_text = ""

        if file:
            session_folder = os.path.join(UPLOAD_FOLDER, session_id)
            os.makedirs(session_folder, exist_ok=True)

            file_path = os.path.join(session_folder, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            file_text = extract_text_from_file(file_path)

        modality_data = process_modalities(
            raw_text=raw_text,
            url=url,
            email_subject=email_subject,
            email_sender=email_sender,
            file_text=file_text
        )

        if not modality_data:
            return JSONResponse(status_code=400, content={"error": "No input provided."})

        features = merge_features(modality_data)

        # -------- LANGUAGE DETECTION -------- #
        combined_text = " ".join([m["content"] for m in modality_data.values()])
        language = detect_language(combined_text)

        # Gap 1 — language-agnostic loop:
        # Pass the detected language into compute_modality_context() so
        # context["script_family"] is populated and flows through to
        # compute_dynamic_weights() in the fusion engine.
        # Before this fix, language was detected but never used in fusion.
        context = compute_modality_context(modality_data, language=language)

        vs = VectorStore(session_id)

        # -------- GRAPH -------- #
        rule_score, reasoning_graph = build_reasoning_graph(features, modality_data)

        # Apply cross-modal reasoning BEFORE fusion
        reasoning_graph = apply_cross_modal_reasoning(reasoning_graph, modality_data)

        # -------- FUSION -------- #
        # context now carries script_family — fusion weights adapt to language
        final_score, verdict, debug_info = fuse_scores(
            reasoning_graph, features, context, vs
        )

        # -------- STORE — structured metadata for typed memory retrieval -------- #
        graph_summary = (
            "LANGUAGE: " + language +
            " | SCRIPT: " + context["script_family"] +
            " | VERDICT: " + verdict +
            " | SCORE: " + str(final_score) +
            " | CATEGORIES: " +
            " ".join(f"{k}:{round(v,1)}" for k, v in sorted(reasoning_graph.get("categories", {}).items())) +
            " | INDICATORS: " +
            " ".join(sorted(i["indicator"] for i in reasoning_graph.get("indicators", [])))
        )

        structured_metadata = {
            "verdict":        verdict,
            "score":          final_score,
            "language":       language,
            "script_family":  context["script_family"],
            "is_non_latin":   context["is_non_latin"],
            "categories":     reasoning_graph.get("categories", {}),
            "indicators":     [i["indicator"] for i in reasoning_graph.get("indicators", [])],
            "num_modalities": context.get("num_modalities", 1),
            "fired_rules":    reasoning_graph.get("fired_cross_modal_rules", []),
        }

        vs.add_texts([graph_summary], metadatas=[structured_metadata])

        report = f"""
Language: {language}
Script family: {context['script_family']} (non-Latin: {context['is_non_latin']})
Final Risk Score: {final_score}
Verdict: {verdict}

Reasoning Graph:
{json.dumps(reasoning_graph, indent=2)}

Context:
{json.dumps(context, indent=2)}

Debug:
{json.dumps(debug_info, indent=2)}
"""
        memory_manager.add_message(session_id, "system", report)

        explanation = generate_explanation(session_id, report, rag_query=graph_summary)

        return {
            "language":       language,
            "script_family":  context["script_family"],
            "risk_score":     final_score,
            "verdict":        verdict,
            "reasoning_graph": reasoning_graph,
            "context":        context,
            "debug":          debug_info,
            "explanation":    explanation,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})