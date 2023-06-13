import pyqrcode
from PIL import Image
import zbarlight
import numpy as np
from matplotlib import cm

class QRProcessor:

    def generate(self, data, image_location):
        qr = pyqrcode.create(data)
        qr.png(image_location, scale=6)

    def decode(self, image):
        img = Image.fromarray(image)
        img.load()
        codes = zbarlight.scan_codes(['qrcode'], img)
        if codes is None or len(codes) == 0:
            return -1
        code = codes[0].decode('utf-8')
        return int(code)