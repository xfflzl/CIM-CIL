# import cv2
import math
import numpy as np
from PIL import Image

class BbxBlur(object):
    def __init__(self, compress_ratio):
        self.compress_ratio = compress_ratio

#    def __call__(self, img, bbx):
#        img = np.array(img)
#        original_size = img.shape[:2]
#        compress_size = (int(original_size[0] / math.sqrt(self.compress_ratio)), int(original_size[1] / math.sqrt(self.compress_ratio)))
#        img_resized = cv2.resize(img, compress_size[::-1], interpolation=cv2.INTER_NEAREST)
#        img_resized = cv2.resize(img_resized, original_size[::-1], interpolation=cv2.INTER_NEAREST)
#        img_resized[bbx[0]:bbx[1], bbx[2]:bbx[3], :] = img[bbx[0]:bbx[1], bbx[2]:bbx[3], :]
#        bbx_rate = (bbx[1] - bbx[0]) * (bbx[3] - bbx[2]) / original_size[0] / original_size[1]
#        compress_rate = bbx_rate + (1.0 - bbx_rate) / self.compress_ratio
#        return Image.fromarray(img_resized), compress_rate
    def __call__(self, img, bbx):
        original_width, original_height = img.width, img.height
        compressed_width, compressed_height = int(original_width / math.sqrt(self.compress_ratio)), int(original_height / math.sqrt(self.compress_ratio))
        img_resized = img.resize((compressed_width, compressed_height), Image.NEAREST).resize((original_width, original_height), Image.NEAREST)
        img_resized.paste(img.crop((bbx[2], bbx[0], bbx[3], bbx[1])), (bbx[2], bbx[0], bbx[3], bbx[1]))
        bbx_rate = (bbx[1] - bbx[0]) * (bbx[3] - bbx[2]) / original_width / original_height
        compress_rate = bbx_rate + (1.0 - bbx_rate) / self.compress_ratio
        return img_resized, compress_rate