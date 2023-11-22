import unittest

import cv2
from imgcat import imgcat
from modelscope import Tasks

from facechain.data_process.preprocessing import get_mask_head
from modelscope.pipelines import pipeline


class TestInference(unittest.TestCase):
    def setUp(self):
        self.segmentation_pipeline = pipeline(Tasks.image_segmentation,
                                              'damo/cv_resnet101_image-multiple-human-parsing', model_revision='v1.0.1')

    def test_inference(self):
        result = self.segmentation_pipeline('assets/user-images/asian_woman_0.jpeg')
        mask_head = get_mask_head(result)
        im = cv2.imread('assets/user-images/asian_woman_0.jpeg')
        im = im * mask_head + 255 * (1 - mask_head)
        imgcat(im)


if __name__ == '__main__':
    unittest.main()
