
from stiv_compute_routine import ifft_img_dir

test_data_dir = "data\datatest\highspeed"





def ifftimgPreprocess(ifft_img):
    image = ifft_img
    height, width, _ = image.shape
    center_x = width // 2
    center_y = height // 2
    crop_size = 2**9
    start_x = center_x - crop_size // 2
    start_y = center_y - crop_size // 2
    cropped_image = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    return cropped_image


if __name__ == "__main__":
    get_test_data_ifft(test_data_dir)
