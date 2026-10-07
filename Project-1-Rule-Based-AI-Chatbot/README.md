# DecodeLabs AI Internship — Project 1
## Rule-Based AI Chatbot 

A deterministic, rule-based chatbot built in Python as the foundation project 
of the DecodeLabs Artificial Intelligence Internship (Batch 2026).

# Objective
Master control flow, decision-making logic, and basic AI concepts by building 
a chatbot that responds to predefined user inputs using a dictionary (hash map) 
instead of an if-elif ladder.

# Features
- Handles greetings, small talk, and exit commands
- Input sanitization (`.lower().strip()`) for case/whitespace tolerance
- Dictionary-based O(1) lookup for instant responses
- Fallback response for unrecognized inputs
- Continuous loop with a clean kill command

# Concepts Applied (from Project PDF)
- **IPO Model**: Input → Process → Output
- **Infinite Loop** with `break` kill command
- **Hash Map** lookup vs If-Elif ladder (O(1) vs O(n))
- **`.get()` method**: atomic lookup + fallback
- **White-Box AI**: fully traceable, zero hallucination risk

# How to Run
```bash
python chatbot.py
