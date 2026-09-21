"""Conversational wrapper around detector.py.
Asks what the user lost, acknowledges it naturally, then runs the existing
detector pipeline (folder scan -> cosine ranking -> Grad-CAM) unchanged.
Run: python chatbot.py   (then type 'quit' to exit)
"""
import sys

import detector  # importing detector runs the ResNet18 setup once, up front

# Cosmetic lead-in cleanup only (NOT real parsing): just trims common phrases
# so "I lost my red backpack" reads as "red backpack" in our replies.
PREFIXES = ["i lost my ", "i lost ", "lost my ", "i'm looking for ",
            "looking for ", "my ", "a ", "the "]

def clean(text):
    """Repeatedly strip known lead-in phrases; fall back to raw text."""
    t = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for p in PREFIXES:
            if t.startswith(p):
                t = t[len(p):]
                changed = True
    return t if t else text.strip()

print("Hi! I'm your Lost and Found assistant. Type 'quit' (or Ctrl+C) to exit.")
while True:
    try:
        item = input("\nWhat did you lose? ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye — good luck finding it!")
        break
    if not item:
        continue
    if item.lower() in {"quit", "exit", "q"}:
        print("Good luck finding it! Bye.")
        break
    name = clean(item)
    print(f"\nSorry to hear about your {name}! Let me scan this folder for candidates...")
    try:
        detector.main()  # same pipeline and output as `python detector.py`
    except SystemExit as e:  # detector.main() exits if no/insufficient images
        print(f"({e}) Put some .jpg/.png images in this folder and try again.")
        continue
    print(f"\nFingers crossed one of these is your {name}!")
