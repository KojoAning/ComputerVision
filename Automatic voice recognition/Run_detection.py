 #import the necessary libraries
from PIL import Image, ImageTk
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import librosa
import gradio as gr
from scipy.io.wavfile import write

#function to process audio file
def process_audio(audio_path):
    audio, fs = librosa.load(audio_path)
    D = librosa.amplitude_to_db(librosa.stft(audio), ref=np.max)
    # Save the spectrogram
    plt.figure(figsize=(10, 5))
    plt.axis('off')
    librosa.display.specshow(D, sr=fs, x_axis='time', y_axis='linear',cmap='viridis')
    plt.savefig('output.png', dpi=300, bbox_inches='tight', pad_inches=0,transparent=True)


def predict():                 
    img = cv.imread("output.png")                                            
    img = cv.resize(img,(300,300))                                           
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)                                 
    img = np.expand_dims(img,0)                                              
    img = np.expand_dims(img,-1)
    final_model = tf.keras.models.load_model('my_model.h5')                           
    prediction = final_model.predict(img)                                             
    prediction = int(prediction[0][0])                                                

    if prediction == 1:
        return "voice matched"
    else:
        return "voice did not match"

def final(audio_path):
    sample_rate, audio_data = audio_path
    output_file = "processed_audio.wav"
    write(output_file, sample_rate, np.array(audio_data, dtype=np.int16))
    process_audio("processed_audio.wav")
    result = predict()
    return result

demo = gr.Interface(final,'audio','text')
demo.launch()