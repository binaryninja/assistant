import openai
import sounddevice as sd
import audiofile as af
from scipy.io.wavfile import write
from gtts import gTTS
import numpy as np
import multiprocessing
from transformers import AutoProcessor, BarkModel
import scipy.io.wavfile
import sounddevice as sd
import numpy as np
import keyboard
import threading
import nltk  # we'll use this to split into sentences
import queue



from bark.generation import (
    generate_text_semantic,
    preload_models,
)
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE


audio_queue = queue.Queue()

preload_models()

# Load Bark
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")
model.to('cuda')


def audio_producer(text, SPEAKER):
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        audio_queue.put(audio_array)  # Put generated audio in the queue
        # Add a quarter second of silence after each sentence
        silence = np.zeros(int(0.25 * SAMPLE_RATE))
        audio_queue.put(silence)

def audio_consumer():
    while True:
        audio_array = audio_queue.get()  # Get audio from the queue
        if audio_array is None:
            break  # None is used as a signal to stop the consumer
        sd.play(audio_array, samplerate=SAMPLE_RATE)
        sd.wait()
        audio_queue.task_done()

def say(text):
    SPEAKER = "v2/en_speaker_6"

    # Start the producer thread
    producer_thread = threading.Thread(target=audio_producer, args=(text, SPEAKER))
    producer_thread.start()

    # Start the consumer thread
    consumer_thread = threading.Thread(target=audio_consumer)
    consumer_thread.start()

    # Wait for the producer thread to finish
    producer_thread.join()

    # Signal the consumer to stop
    audio_queue.put(None)
    consumer_thread.join()


# def say(text):
#     # Split text into sentences
#     sentences = nltk.sent_tokenize(text)
#     SPEAKER = "v2/en_speaker_6"
#     silence = np.zeros(int(0.25 * SAMPLE_RATE))  # Quarter second of silence

#     for sentence in sentences:
#         # Generate audio for each sentence
#         audio_array = generate_audio(sentence, history_prompt=SPEAKER)

#         # Play the sentence in a separate thread
#         play_thread = threading.Thread(target=play_audio, args=(audio_array,))
#         play_thread.start()

#         # Generate the next sentence while the current one is playing
#         if sentences.index(sentence) < len(sentences) - 1:
#             next_sentence = sentences[sentences.index(sentence) + 1]
#             next_audio_array = generate_audio(next_sentence, history_prompt=SPEAKER)

#         # Wait for the current sentence to finish playing
#         play_thread.join()

#         # Play silence
#         sd.play(silence, samplerate=SAMPLE_RATE)
#         sd.wait()


def record_audio(filename, sec, sr = 44100):
    audio = sd.rec(int(sec * sr), samplerate=sr, channels=2, blocking=False)
    sd.wait()
    write(filename, sr, audio)

def record_audio_manual(filename, sr = 44100):
    input("  ** Press enter to start recording **")
    audio = sd.rec(int(10 * sr), samplerate=sr, channels=2)
    input("  ** Press enter to stop recording **")
    sd.stop()
    write(filename, sr, audio)

def play_audio(audio_array):
    sd.play(audio_array, samplerate=SAMPLE_RATE)
    sd.wait()

def transcribe_audio(filename):
    audio_file= open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    return transcript

def translate_audio(filename):
    audio_file= open(filename, "rb")
    translation = openai.Audio.translate("whisper-1", audio_file)
    audio_file.close()
    return translation

def save_text_as_audio(text, audio_filename):
    myobj = gTTS(text=text, lang='en', slow=False)  
    myobj.save(audio_filename)



