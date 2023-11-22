import unittest

from facechain.data_process.preprocessing import Blipv2


class TestInference(unittest.TestCase):
    def setUp(self):
        self.blipv2 = Blipv2()

    def test_inference(self):
        result = self.blipv2.extract_face_info('assets/theme-images', imname='pregnent_01.png', debug=True)
        print(result)


if __name__ == '__main__':
    unittest.main()
