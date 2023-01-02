import json
from a2w.audio_handler import audio_to_words,AudioToWords
from a2w.micro_handler import listen_to_mic,StreamToWords
micro = StreamToWords('./a2w/model')
audio = AudioToWords('./a2w/model')
