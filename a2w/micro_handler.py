from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import wave
import pyaudio
import numpy as np

class StreamToWords:
    def __init__(self,model_path):
        SetLogLevel(0)
        if not os.path.exists(model_path):
            print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
            exit (1)

        self.model = Model(model_path)

    def realtime_to_word(self,data,frame_rate,frame_size,time_rec):
        rec = KaldiRecognizer(self.model, frame_rate)
        rec.SetMaxAlternatives(10)
        rec.SetWords(True)
        for i in range(0, int(frame_rate / frame_size * time_rec)-1):
            chunk  = data[i*frame_size:(i+1)*frame_size]
            rec.AcceptWaveform(chunk) 
        return rec.FinalResult()


def listen_to_mic(worker,save=False,save_path=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)



    print("* recording")
    
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    if save:
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        

    frames = np.array(frames)
    data = frames.tobytes()
    RATE
    CHUNK
    RECORD_SECONDS
    result = worker.realtime_to_word(data,RATE,CHUNK,RECORD_SECONDS)
    return result

