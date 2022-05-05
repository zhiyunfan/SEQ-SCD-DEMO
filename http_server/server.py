from matplotlib.font_manager import json_load
from websocket_server import WebsocketServer
import soundfile as sf
import threading
import base64
import time
import json
import numpy as np
import warnings
from docopt import docopt
from pathlib import Path
from tornado.options import define, options
import torch
from base import Application
import argparse


def main(client, server, message):
    data_dict = json.loads(message)
    wav = data_dict["wav"]
    wav = bytes2wav(wav)
    params['wav'] = wav
    params['sr'] = 16000
    params['best_threshold'] = data_dict["threshold"]
    results = app.run(**params)
    message_back=json.dumps({'results':"%s"%results})
    server.send_message_to_all(message_back)

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

def make_parser():
    parser = argparse.ArgumentParser()
    # TODO: update version automatically
    parser.add_argument("--batch",
                         type=int,
                         help="",
                         default=128)
    parser.add_argument("--step",
                         type=float,
                         help="",
                         default=0.25)
    parser.add_argument("--down_rate",
                         type=int,
                         help="",
                         default=8)
    parser.add_argument("--best_threshold",
                         type=float,
                         help="",
                         default=0.1)

    args = parser.parse_args()
    params = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #train_dir = Path(arg["<train>"]).expanduser().resolve(strict=True)
    #app = Application.from_train_dir(train_dir, training=False)
    app = Application()

    params["batch_size"] = args.batch
    params["step"] = args.step
    params["down_rate"] = args.down_rate
    params["best_threshold"] = args.best_threshold
    params["device"] = torch.device(device)
    return params, app

if __name__ == '__main__':

    params, app = make_parser()

    # wav, sr = sf.read('/mnt/lustre/xushuang2/zyfan/program/code/SEQ-SCD/http/IB4005.Mix-Headset.wav')
    # # vp_wav,_ = sf.read('./eval_input/002CJ-EN.wav')
    # # target_wav = model.run(mix_wav, vp_wav)
    # params['wav'] = wav
    # params['sr'] = sr
    # results = app.run(**params)
    # print('done', results)
    server = WebsocketServer("0.0.0.0", 9000)
    server.set_fn_message_received(main)
    server.run_forever()
    
