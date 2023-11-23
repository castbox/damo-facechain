import unittest

import cv2
import numpy
from PIL import Image
from imgcat import imgcat
from modelscope import pipeline, Tasks
from modelscope.outputs import OutputKeys

from facechain.data_process.preprocessing import Blipv2


class TestInference(unittest.TestCase):
    def setUp(self):
        self.blipv2 = Blipv2()

    def test_inference(self):
        result = self.blipv2.extract_face_info('assets/user-images', imname='black_man_0.jpeg', debug=True)
        print(result)

        image_face_fusion = pipeline(Tasks.image_face_fusion,
                                     model='damo/cv_unet-image-face-fusion_damo')
        template = Image.open('assets/theme-images/black_man_fused.png')
        user_path = result

        for i in range(3):
            result = image_face_fusion(dict(template=template, user=user_path))
            output = cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB)
            imgcat(output)
            template = Image.fromarray(output)

        imgcat(cv2.cvtColor(result[OutputKeys.OUTPUT_IMG], cv2.COLOR_BGR2RGB))
        print('finished!')


if __name__ == '__main__':
    unittest.main()
