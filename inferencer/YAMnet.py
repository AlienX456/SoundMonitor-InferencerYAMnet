import scipy.signal as ss
from inferencer.Inferencer import Inferencer
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from scipy.io import wavfile


class YAMnet(Inferencer):

    def __init__(self):
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')

        class_map_path = self.model.class_map_path().numpy()

        self.class_names = []
        with tf.io.gfile.GFile(class_map_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.class_names.append(row['display_name'])

    def runInferencer(self, buffer, filename):
        sample_rate, wav_data = wavfile.read(buffer)
        print('sample -- ', type(sample_rate))
        sample_rate, wav_data = self.ensure_sample_rate(sample_rate, wav_data)
        waveform = wav_data / tf.int16.max
        scores, embeddings, spectrogram = self.model(waveform)
        scores_np = scores.numpy()
        scores_mean = scores_np.mean(axis=0)
        top_n_index = scores_mean.argsort()[-5:][::-1]
        top_n_classes = [self.class_names[index] for index in top_n_index]
        top_n_scores = [scores_mean[index] for index in top_n_index]

        result = {"audio_filename": filename}

        for i in range(0, len(top_n_classes)):
            result[top_n_classes[i]] = top_n_scores[i].item()

        return result

    def ensure_sample_rate(self, original_sample_rate, waveform,
                           desired_sample_rate=16000):

        if original_sample_rate != desired_sample_rate:
            desired_length = int(round(float(len(waveform)) /
                                       original_sample_rate * desired_sample_rate))
            waveform = ss.resample(waveform, desired_length)
        return desired_sample_rate, waveform
