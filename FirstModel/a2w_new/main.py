from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import wave
import json

class AudioToWords:
    def __init__(self, audio_path, model_path):
        
        self.audio_path = audio_path
        self.model_path = model_path

        SetLogLevel(0)

        if not os.path.exists(model_path):
            print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit (1)

        self.wf = wave.open(audio_path, "rb")
        if self.wf.getnchannels() != 1 or self.wf.getsampwidth() != 2 or self.wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format mono PCM.")
            exit (1)

    def get_words(self):
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, self.wf.getframerate())
        rec.SetMaxAlternatives(10)
        rec.SetWords(True)

        while True:
            data = self.wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        return rec.FinalResult()

audio_path = 'test.wav'
model_path = 'model'

sample = AudioToWords(audio_path, model_path)
result = sample.get_words()

print(json.loads(result))