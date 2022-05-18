import websocket
import threading
import base64
import time
import json
import wave
import queue
import soundfile as sf
import numpy as np

def on_message(ws, message):
    global results
    results = eval(message)["results"]
    queue_signal.put(1)
 
def on_error(ws, error):
    print(error)
 
 
def on_close(ws):
    print('close')
 
def start_websocket():
    global ws
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://172.18.30.121:9000",
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()


def wav2bytes(wav):
    wav=wav*32768
    wav = np.int16(wav)
    wav_bytes = wav.astype(np.int16).tostring()
    wav_bytes_base64 = str(base64.b64encode(wav_bytes), encoding='utf-8')
    return wav_bytes_base64

def bytes2wav(bytes_base64):
    wav_bytes = base64.b64decode(bytes_base64)
    wav = np.frombuffer(buffer=wav_bytes, dtype=np.int16)
    wav=wav/32768
    return wav

#if __name__ == '__main__':
def main(wav_path, threshold):
    global queue_signal 
    queue_signal = queue.Queue()

    wav,_ = sf.read(wav_path)
    wav_bytes = wav2bytes(wav)

    message=json.dumps({'wav':"%s"%wav_bytes, 'threshold':threshold})

    th = threading.Thread(target=start_websocket)
    th.start()
    time.sleep(0.1)

    ws.send(message)
    while queue_signal.empty():
        time.sleep(1)
    while not queue_signal.empty():
        q = queue_signal.get()
    ws.close()
    return results 
