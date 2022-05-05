import streamlit as st
import pyaudio
import wave
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from ws_client import main
from core import notebook
from core.segment import Segment 
from core.timeline import Timeline
import matplotlib.pyplot as plt
from PIL import Image
import datetime


# side bar
image1 = Image.open('logo.jpg')
st.sidebar.image(image1, caption='',
          use_column_width=True)

threshold = st.sidebar.slider(label='threshold', min_value=0.0, max_value=1.0, step=0.01)

d1 = datetime.date.today()
st.sidebar.write('Date:', d1)




FORMAT = pyaudio.paInt16 
CHANNELS = 1
RATE = 16000
CHUNK = 2048
wav_dir = 'wav_dir/'

st.title('Sequence-level Speaker Change Detection')

# slide
#x = st.slider('x')
#st.write(x, 'squared is', x * x)

st.header('Prepare:')
reset = st.button('Reset', key='reset')

col4, col5=st.columns([0.5, 0.5])

with col4:
    dur = st.text_input('Please enter a recording duration (> 4s)', value='10', key=None)
with col5:
    st.empty()    
start_recording = st.button('Start Recording', key='start')

uttid = str(uuid.uuid4())
if dur != '':
    dur = int(dur)
    if dur < 4:
        dur = 4

def time2sec(time):
    print('time:', time)
    time = time.split(':')
    ret = eval(time[0])*3600 + eval(time[1])*60 + eval(time[2])
    return ret

def process_fun(inputs):
    timeline = Timeline()
    inputs = inputs.strip()[1:-1].split('\n')
    for item in inputs:
        item = item.strip()[1:-1].strip().split('-->')
        start = time2sec(item[0])
        end = time2sec(item[1])
        timeline.add(Segment(start, end))
    return timeline

if start_recording:
    start_message = st.empty()
    start_message.write("Processing, please wait...")
    uttid = str(uuid.uuid4())
    p = pyaudio.PyAudio()  # 实例化对象
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)  # 打开流，传入响应参数
    wf = wave.open(wav_dir + uttid+'.wav', 'wb')  # 打开 wav 文件。
    wf.setnchannels(CHANNELS)  # 声道设置
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 采样位数设置
    wf.setframerate(RATE)  # 采样频率设置

    for _ in range(dur * RATE // CHUNK):
        data = stream.read(CHUNK)
        wf.writeframes(data)  # 写入数据
    stream.stop_stream()  # 关闭流
    stream.close()
    p.terminate()
    wf.close()

    wav_file = Path(wav_dir + uttid+'.wav')

    #play = st.button('Play', key='play')
    audio_file = open(wav_file, 'rb')
    audio_bytes = audio_file.read()
    st.header('Replay Recorded Audio:')
    st.audio(audio_bytes, format='audio/ogg')
    results = main(wav_file, threshold)
    ## plot results
    fig, ax = plt.subplots(figsize=(6, 2)) 
    notebook.crop = Segment(0, dur)

    results_plot = process_fun(results)
    _ = notebook.plot_timeline(results_plot)
    print(results_plot)
    col1, col2, col3=st.columns([0.23, 1, 0.14])
    with col1:
        st.empty()
    with col2:
        st.pyplot(fig)
    with col3:
        st.empty()

    # print the segmentation results
    st.header('Segmentation Results:\n' + results)
    st.write('Done !')
    st.balloons()

