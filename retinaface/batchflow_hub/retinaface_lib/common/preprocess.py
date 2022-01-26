import numpy as np
import cv2


# This function is modified from the following code snippet:
# https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L49-L74
def resize_image(img, scales, allow_upscaling):
    img_h, img_w = img.shape[0:2]
    target_size = scales[0]
    max_size = scales[1]

    if img_w > img_h:
        im_size_min, im_size_max = img_h, img_w
    else:
        im_size_min, im_size_max = img_w, img_h

    im_scale = target_size / float(im_size_min)
    if not allow_upscaling:
        im_scale = min(1.0, im_scale)

    if np.round(im_scale * im_size_max) > max_size:
        im_scale = max_size / float(im_size_max)

    if im_scale != 1.0:
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )

    return img, im_scale

def resize_image2(img, scales, allow_upscaling):
    target_size = [1980, 1980]
    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)

    dsize = (
        int(img.shape[1] * factor),
        int(img.shape[0] * factor),
    )
    img = cv2.resize(img, dsize)

    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    top, bottom, left, right = (diff_0 // 2, diff_0 - diff_0 // 2, diff_1 // 2, diff_1 - diff_1 // 2)
    # img2 = np.pad(
    #     img,
    #     (
    #         (diff_0 // 2, diff_0 - diff_0 // 2),
    #         (diff_1 // 2, diff_1 - diff_1 // 2),
    #     ),
    #     "constant",
    # )
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return img, factor, (top, bottom, left, right)



# This function is modified from the following code snippet:
# https://github.com/StanislasBertrand/RetinaFace-tf2/blob/5f68ce8130889384cb8aca937a270cea4ef2d020/retinaface.py#L76-L96
def preprocess_image(img, allow_upscaling, pad):
    pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pixel_scale = float(1.0)
    scales = [1024, 1980]

    if pad:
        img, im_scale, im_pad = resize_image2(img, scales, allow_upscaling)
    else:
        im_pad = None
        img, im_scale = resize_image(img, scales, allow_upscaling)
        
    img = img.astype(np.float32)
    im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

    # Make image scaling + transpose (N,H,W,C) to (N,C,H,W)
    for i in range(3):
        im_tensor[0, :, :, i] = (img[:, :, i] / pixel_scale - pixel_means[i]) / pixel_stds[i]

    return im_tensor, img.shape[0:2], im_scale, im_pad