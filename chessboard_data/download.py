from PIL import Image

import io
import requests
import time


def download_image(url, image_file_path):
    r = requests.get(url, timeout=5)
    if r.status_code != requests.codes.ok:
        print(image_file_path)  # There is no left10.jpg in opencv/samples/data/
        return

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)


if __name__ == '__main__':
    base_url = r'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/'
    for i in range(1, 15):
        img_name = 'left' + str(i).zfill(2) + '.jpg'
        download_image(base_url+img_name, img_name)
        time.sleep(1)  # To prevent from requesting too frequently

    print("download successful")
