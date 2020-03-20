from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage import transform


class ImageUtils:

    @staticmethod
    def get_image(img_path, img_size=False):
        img = load_img(img_path)
        img = img_to_array(img, dtype=np.float32)
        if img_size != False:
            img = transform.resize(img, img_size, preserve_range=True)
        return img

    @staticmethod
    def resize_img(img, size):
        if len(size) == 2:
            size += 3
        return transform.resize(img, size, preserve_range=True)