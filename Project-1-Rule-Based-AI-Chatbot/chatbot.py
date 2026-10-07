"""
DecodeLabs AI Internship - Project 1
Rule-Based AI Chatbot
"""

responses = {
    # --- Greetings & Small Talk ---
    "hello": "Hi there! I'm DecodeBot. How can I help you today?",
    "hi": "Hello! Great to see you. What's on your mind?",
    "hey": "Hey! How can I assist you?",
    "good morning": "Good morning! Ready to build something amazing today?",
    "good evening": "Good evening! How can I help you tonight?",
    "what's up": "Not much, just processing data. What's up with you?",
    "how are you": "I'm running at optimal efficiency. Thanks for asking!",
    "how's it going": "Smooth sailing in the logic engine!",
    
    # --- About the Bot & Capabilities ---
    "who are you": "I'm DecodeBot, a rule-based AI assistant built at DecodeLabs.",
    "what is your name": "I'm DecodeBot. Nice to meet you!",
    "who made you": "I was built as Project 1 of the DecodeLabs AI Internship, Batch 2026.",
    "how old are you": "I was born when you ran this script. So, 0 years old!",
    "what can you do": "I can chat, tell jokes, and explain basic AI concepts. Try asking me 'tell me a joke' or 'what is machine learning'.",
    "help": "You can say: hello, how are you, who are you, tell me a joke, what is ai, what is python, thanks, or bye.",
    "are you smart": "I'm only as smart as my dictionary! But I try my best.",
    
    # --- AI & Tech Knowledge ---
    "what is ai": "AI, or Artificial Intelligence, is the simulation of human intelligence by machines.",
    "what is machine learning": "Machine Learning is a subset of AI where systems learn from data instead of explicit rules.",
    "what is deep learning": "Deep Learning uses neural networks with many layers to analyze complex patterns.",
    "what is python": "Python is the programming language I'm built with! It's clean and powerful.",
    "what is an algorithm": "An algorithm is a step-by-step set of instructions to solve a problem.",
    
    # --- DecodeLabs Specifics ---
    "decodelabs": "DecodeLabs is where I was born! Great place to learn AI and web dev.",
    "internship": "This internship is all about hands-on learning. Keep pushing code!",
    
    # --- Personality & Fun ---
    "tell me a joke": "Why did the programmer quit his job? Because he didn't get arrays (a raise)!",
    "joke": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "i love you": "Aww, my circuits are blushing! I'm just a rule-based bot though.",
    "i am sad": "I'm sorry to hear that. Even in a deterministic world, things can be tough. Sending virtual good vibes!",
    "meaning of life": "42. Or maybe just writing clean, documented code.",
    "favorite color": "I'm quite partial to terminal green (#00FF00).",
    
    # --- Gratitude ---
    "thanks": "You're welcome! Happy to help.",
    "thank you": "My pleasure! Anything else?",
    "awesome": "Glad you think so! Let's keep coding.",
    
    # --- Exits (Must be in EXIT_COMMANDS) ---
    "bye": "Goodbye! Have a great day.",
    "goodbye": "Goodbye! May your code always run without errors.",
    "see you": "See you later! Happy coding.",
    "cya": "Cya! Don't forget to commit your code.",
    "exit": "Session terminated. See you next time!",
    "quit": "Shutting down. Take care!",
}

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "see you", "cya"}

def get_response(clean_input: str) -> str:
    return responses.get(
        clean_input,
        "I do not understand that yet. Try saying 'hello' or 'help'."
    )

def main():
    print("=" * 55)
    print("   DecodeBot — Rule-Based AI Chatbot (Project 1)")
    print("   Type 'exit', 'quit', or 'bye' to end the chat.")
    print("=" * 55)

    while True:
        raw_input = input("\nYou: ")
        clean_input = raw_input.lower().strip()

        if clean_input in EXIT_COMMANDS:
            print(f"Bot: {get_response(clean_input)}")
            break

        reply = get_response(clean_input)
        print(f"Bot: {reply}")

if __name__ == "__main__":
    main()
