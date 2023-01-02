from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import wave
import json

class AudioToWords:
    def __init__(self, model_path):
        
        self.model_path = model_path

        SetLogLevel(0)

        if not os.path.exists(model_path):
            print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit (1)

        

    def get_words(self,wf):
        model = Model(self.model_path)
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetMaxAlternatives(10)
        rec.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec.AcceptWaveform(data)

        return rec.FinalResult()


def read_wf(audio_path):
    wf = wave.open(audio_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print ("Audio file must be WAV format mono PCM.")
        exit (1)
    return wf

def audio_to_words(audio_path,worker):
    wf = read_wf(audio_path)

    return worker.get_words(wf)


