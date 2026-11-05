import sys
import pygame
import pyttsx3
import speech_recognition as sr
import ollama
import subprocess
import face_recognition
import face_recognition_models
import cv2
import numpy as np
import os
import time
import warnings
from pyvidplayer2 import Video

WIDTH = 800
HEIGHT = 600
MARGIN = 40
#------------------------------------------------------------



# Must be set BEFORE importing cv2: stops the slow MSMF hardware-transform
# startup that can make the camera take a minute+ to open on Windows.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"



warnings.filterwarnings("ignore", category=UserWarning)

t0 = time.time()
#jarvis = Video("jarvis.mp4")

print(f"imports: {time.time() - t0:.1f}s")

# name -> image file. Add more people here.
PEOPLE = {
    "Shailen": "shailen.jpg",
    "Shaan": "shaan.jpg",
    "Sohin": "sohin.jpg",
    "Rikin": "rikin.jpg",
    "Reema": "reema.jpg"

}

known_face_encodings = []
known_face_names = []

t0 = time.time()
for name, path in PEOPLE.items():
    try:
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"No face found in {path}, skipping {name}.")
            continue
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
    except Exception as e:
        print(f"Could not load {path}: {e}")
print(f"encode known faces: {time.time() - t0:.1f}s")

if not known_face_encodings:
    raise SystemExit("No known faces loaded. Check your image files.")

t0 = time.time()
# CAP_DSHOW opens the camera far faster than the default backend on Windows.
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"open camera: {time.time() - t0:.1f}s")

SCALE = 0.25  # detect on a quarter-size frame for speed
frame_count = 0
face_locations = []
face_names = []
detecting = True

while detecting:
    ret, frame = video_capture.read()
    if not ret:
        print("Could not read from camera.")
        break

    # Only run detection on every other frame.
    if frame_count % 2 == 0:
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)  # contiguous, dlib-safe

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best = int(np.argmin(distances))
            if matches[best]:
                name = known_face_names[best]
            face_names.append(name)
    frame_count += 1

    inv = int(1 / SCALE)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = top * inv, right * inv, bottom * inv, left * inv
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        if name in PEOPLE:
            print(f"Recognized {name}!")
            detecting = False
            video_capture.release()
            cv2.destroyAllWindows()
            running = True
            
            
        
    cv2.imshow("Video", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()





#------------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("My Pygame Window")
r = sr.Recognizer()

engine = pyttsx3.init()
voices = engine.getProperty("voices")

david = next(
    (voice for voice in voices if "David" in voice.name),
    voices[0] if voices else None,
)

if david:
    engine.setProperty("voice", david.id)

engine.setProperty("rate", 190)
engine.setProperty("volume", 1.0)

engine.say("System initialized.")
engine.runAndWait()

result = ollama.generate(
    model="llama3.2",
    prompt="",
)
response = str(result["response"])


font = pygame.font.SysFont("monospace", 24)
clock = pygame.time.Clock()
#running = True
spoken = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    # Wrap lines to fit the current window width.
    max_width = max(1, screen.get_width() - MARGIN * 2)
    lines = []

    for paragraph in response.splitlines() or [""]:
        words = paragraph.split()
        current_line = ""

        for word in words:
            candidate = f"{current_line} {word}".strip()

            if font.size(candidate)[0] <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)
        elif not words:
            lines.append("")

    line_height = font.get_linesize()
    y = max(MARGIN, (screen.get_height() - len(lines) * line_height) // 2)

    for line in lines:
        text_surface = font.render(line, True, (255, 255, 255))
        screen.blit(text_surface, (MARGIN, y))
        y += line_height

    pygame.display.flip()
    if not spoken:
        print("Speaking response...")
        engine.say(response)
        #engine.say("hello")
        #engine.runAndWait()
        spoken = True
    clock.tick(30)

    #------------------------------------------------------------
    history = []
    global instructions
#   instructions = """
#  You are Jarvis, a concise, polished voice assistant. Address the user as "sir".
#Reply in one short sentence. If you don't know, give your best guess.
#If the request is unclear, say "Sir, please repeat that."

#Examples:
#User: what's the capital of France
#Jarvis: Paris, sir.
#User: what's two plus two
#Jarvis: Four, sir."""
    instructions = "You are Jarvis, just act like him"
    try:
        with sr.Microphone() as source:
            print("Listening...")


            
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source)
            text = r.recognize_google(audio)
            text = text.lower()  
            #print("Noise level:", r.energy_threshold)
            print("You said:", text)

            history.append(f"User's Name: {name}")
            history.append(f"INSTRUCTIONS: {instructions}")
            result = ollama.chat(
            model="llama3.2",
            #model="nimble-v3:9b",
            messages=[{"role": "user", "content": f"{text} Instructions: \n\n History: {history}"}],
            #prompt=f"{history} {text} {instructions}",
            #options={"temperature": 0.3} #low is more accurate, high is more creative
            options={"temperature": 0.6}
)
            #response = str(result["response"])
            response = result["message"]["content"]
            engine.say(response)
            history.append(f"User: {text}\nJarvis: {response}\n")
            if "exit" in text:
                print("Exiting program...")
                break
#------------------------------------------------------------
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("Could not understand audio")

    except KeyboardInterrupt:
        print("Program terminated by user")
        break

    try :
        subprocess.Popen(response)
        

    except Exception as e:
        pass
        #print(f"Error: {str(e)}" "or maybe the application is not installed on your system")

    if text.startswith("open"):
        name = text[4:].strip()          # "open notepad" -> "notepad"
        known_applications = {
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "paint": "mspaint.exe",
            "wordpad": "wordpad.exe",
            "cmd": "cmd.exe",
            "files": "explorer.exe",
            "chrome": "chrome.exe",
            "minecraft": "minecraft-launcher.exe",
        }
        if name in known_applications:
            try:
                subprocess.Popen(known_applications[name])
                print("Opening application...")
            except Exception as e:
                print(f"Error opening application: {str(e)}")
        else:
            print(f"Application not found: {name!r}")
        pass
    print("Noise level:", r.energy_threshold)



    #if jarvis.draw(WIDTH, HEIGHT, (0, 0)):
        #pygame.display.update()
pygame.quit()
sys.exit()
