import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import face_recognition
import ollama
import pygame
import pyttsx3
import speech_recognition as sr

WIDTH, HEIGHT, MARGIN = 800, 600, 40
MODEL = "llama3.2"
INSTRUCTIONS = "You are Jarvis, just act like him. Keep replies to one short sentence."

APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "paint": "mspaint.exe",
    "wordpad": "wordpad.exe",
    "cmd": "cmd.exe",
    "files": "explorer.exe",
    "chrome": "chrome.exe",
    "minecraft": "minecraft-launcher.exe",
}

FOLDERS = {
    "downloads": Path.home() / "Downloads",
    "documents": Path.home() / "Documents",
    "desktop": Path.home() / "OneDrive" / "Desktop",
    "jarvis": Path.home() / "OneDrive" / "Desktop" / "jarvis",
    "minecraft": Path.home() / "AppData" / "Roaming" / ".minecraft",
}

state = {"text": "Starting...", "running": True}
history = []

# ---------- face recognition ----------
known_encoding = None
try:
    _img = face_recognition.load_image_file("me.jpg")
    _encs = face_recognition.face_encodings(_img)
    known_encoding = _encs[0] if _encs else None
    if known_encoding is None:
        print("No face found in me.jpg - run pic.py again.")
except Exception as e:
    print("Could not load me.jpg:", e)


def see_me(tries=15):
    if known_encoding is None:
        return None
    cam = cv2.VideoCapture(0)
    found = False
    for _ in range(tries):  # first frames are often dark, so try several
        ok, frame = cam.read()
        if not ok:
            continue
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        for enc in face_recognition.face_encodings(rgb):
            if face_recognition.compare_faces([known_encoding], enc, tolerance=0.5)[0]:
                found = True
        if found:
            break
    cam.release()
    return found


# ---------- folders ----------
def open_folder(name):
    path = FOLDERS.get(name)
    if path and path.exists():
        os.startfile(path)
        return f"Opening {name}, sir."
    return f"I don't have {name} on my list, sir."


def list_folder(name, limit=10):
    path = FOLDERS.get(name)
    if not (path and path.exists()):
        return f"I don't have {name} on my list, sir."
    items = sorted(p.name for p in path.iterdir())[:limit]
    return f"{name} contains: " + ", ".join(items) if items else f"{name} is empty, sir."


# ---------- voice loop ----------
def voice_loop():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    david = next((v for v in voices if "David" in v.name), voices[0] if voices else None)
    if david:
        engine.setProperty("voice", david.id)
    engine.setProperty("rate", 190)
    engine.setProperty("volume", 1.0)

    def say(msg):
        state["text"] = msg
        engine.say(msg)
        engine.runAndWait()

    r = sr.Recognizer()
    say("System initialized.")

    while state["running"]:
        try:
            with sr.Microphone() as source:
                state["text"] = "Listening..."
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, phrase_time_limit=15)
            text = r.recognize_google(audio).lower()
        except sr.UnknownValueError:
            state["text"] = "Could not understand audio"
            continue
        except sr.RequestError as e:
            state["text"] = f"Speech service error: {e}"
            continue
        except Exception as e:
            state["text"] = f"Mic error: {e}"
            break

        print("You said:", text)

        if "exit" in text:
            say("Goodbye, sir.")
            state["running"] = False
            break

        if "who am i" in text or "do you see me" in text:
            state["text"] = "Looking..."
            seen = see_me()
            if seen is None:
                say("I have no photo of you on file, sir.")
            elif seen:
                say("Welcome back, sir.")
            else:
                say("I don't recognize you, sir.")
            continue

        if text.startswith("open ") and text.endswith(" folder"):
            say(open_folder(text[5:-7].strip()))
            continue

        if text.startswith("list ") and text.endswith(" folder"):
            say(list_folder(text[5:-7].strip()))
            continue

        if text.startswith("open"):
            name = text[4:].strip()
            target = APPS.get(name)
            if target:
                try:
                    subprocess.Popen(target)
                    say(f"Opening {name}, sir.")
                except Exception as e:
                    print("Open error:", e)
                    say(f"I couldn't open {name}, sir.")
            else:
                say(f"I don't have {name} on my list, sir.")
            continue

        state["text"] = "Thinking..."
        messages = [{"role": "system", "content": INSTRUCTIONS}] + history[-8:]
        messages.append({"role": "user", "content": text})
        try:
            result = ollama.chat(
                model=MODEL,
                messages=messages,
                options={"temperature": 0.6},
            )
            response = result["message"]["content"].strip()
        except Exception as e:
            print("MODEL ERROR:", repr(e))
            say("Model error, sir.")
            continue

        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response})
        say(response)


# ---------- window ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Jarvis")
font = pygame.font.SysFont("monospace", 24)
clock = pygame.time.Clock()

threading.Thread(target=voice_loop, daemon=True).start()


def wrap(text, max_width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for word in paragraph.split():
            candidate = f"{current} {word}".strip()
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        lines.append(current)
    return lines


while state["running"]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state["running"] = False

    screen.fill((0, 0, 0))
    lines = wrap(state["text"], max(1, screen.get_width() - MARGIN * 2))
    lh = font.get_linesize()
    y = max(MARGIN, (screen.get_height() - len(lines) * lh) // 2)
    for line in lines:
        screen.blit(font.render(line, True, (255, 255, 255)), (MARGIN, y))
        y += lh
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()