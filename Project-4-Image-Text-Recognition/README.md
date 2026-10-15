# DecodeLabs AI Internship — Project 4 (Optional)
## Image & Text Recognition — OCR Pipeline 

An end-to-end OCR (Optical Character Recognition) pipeline built with Google's 
Tesseract engine, as the optional mastery milestone of the DecodeLabs 
Artificial Intelligence Internship (Batch 2026).

# Objective
Integrate a pre-trained AI library into a functional vision workflow — applying 
industry-standard pre-processing and enforcing an 80% confidence threshold on 
the final output.

# Features
- **3-step pre-processing pipeline** (per project spec):
  1. Grayscale conversion — collapses RGB → intensity
  2. Gaussian blur — eliminates micro-noise
  3. Adaptive thresholding (Otsu's method) — forces binary decisions
- **OCR extraction** via `pytesseract` (Google's Tesseract 5 engine, LSTM-based)
- **Multi-variant testing** — automatically picks the best pre-processing output
- **Confidence gating** — accepts only tokens ≥ 80% (per project requirement)
- **Bounding-box visualization** — annotates the original image with detections
- **Self-contained** — auto-generates a sample image if none exists

# Concepts Applied (from Project PDF)
- **The Perception Matrix** — chose Path 1 (OCR) over Path 2 (Object Detection)
- **IPO Model** — Input (image) → Process (pre-processing + Tesseract) → Output (strings + boxes)
- **Transfer Learning** — using Google's pre-trained OCR model, no training required
- **Pre-processing integrity** — grayscale, blur, adaptive threshold (all 3 demonstrated)
- **80% Confidence Gate** — filters confident hallucinations
- **Softmax output interpretation** — displays per-token confidence percentages

# Results

| Metric | Value |
|---|---|
| Tokens detected | 12 |
| Tokens passing 80% gate | 12 (100%) |
| Average confidence | 94.7% |
| Extracted string accuracy | 100% |

**Extracted text:**
