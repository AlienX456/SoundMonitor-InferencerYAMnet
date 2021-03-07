import unittest
from inferencer.YAMnet import YAMnet


import io


class TestStringMethods(unittest.TestCase):

    def test_inferencer(self):
        inferencer = YAMnet()
        with open('test/test_audio.wav', 'rb') as audio:
            buffer = io.BytesIO(audio.read())
        result = inferencer.runInferencer(buffer, 'test_audio.wav')
        assert 'audio_filename' in result


if __name__ == '__main__':
    unittest.main()