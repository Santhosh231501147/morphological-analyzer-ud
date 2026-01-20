from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os
import spacy

# Add src directory to path
sys.path.append(os.path.dirname(__file__))
from ud_loader import load_conllu
from evaluation import evaluate_pos_accuracy, build_confusion_matrix
from exporter import export_user_input_analysis, export_ud_dataset_analysis

app = FastAPI(title="Morphological Analyzer API", version="1.0.0")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    sentences: List[str]


class TokenAnalysis(BaseModel):
    FORM: str
    LEMMA: str
    UPOS: str
    HEAD: int
    DEPREL: str


class SentenceAnalysis(BaseModel):
    sentence: str
    tokens: List[TokenAnalysis]


def analyze_text(text: str) -> SentenceAnalysis:
    """Analyze text using spaCy and return morphological analysis."""
    doc = nlp(text)
    
    tokens = []
    for i, token in enumerate(doc):
        token_analysis = TokenAnalysis(
            FORM=token.text,
            LEMMA=token.lemma_,
            UPOS=token.pos_,
            HEAD=token.head.i + 1,  # Convert to 1-based indexing
            DEPREL=token.dep_
        )
        tokens.append(token_analysis)
    
    return SentenceAnalysis(sentence=text, tokens=tokens)


@app.post("/analyze", response_model=SentenceAnalysis)
async def analyze_endpoint(input_data: TextInput):
    """Analyze a single sentence."""
    try:
        return analyze_text(input_data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/evaluate")
async def evaluate_endpoint():
    """Evaluate POS accuracy on UD dataset."""
    try:
        dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
        results = evaluate_pos_accuracy(dataset_path)
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UD dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/confusion-matrix")
async def confusion_matrix_endpoint():
    """Get confusion matrix for POS tagging."""
    try:
        dataset_path = "dataset/UD_English-EWT/en_ewt-ud-train.conllu"
        matrix = build_confusion_matrix(dataset_path)
        return matrix
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UD dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Confusion matrix generation failed: {str(e)}")


@app.post("/batch", response_model=List[SentenceAnalysis])
async def batch_analyze_endpoint(input_data: BatchInput):
    """Analyze multiple sentences."""
    try:
        results = []
        for sentence in input_data.sentences:
            analysis = analyze_text(sentence)
            results.append(analysis)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
