import os

import cv2
import numpy as np


def dilate(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def erode(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def main(src: str, out: str):
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(src)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=16,
        detectShadows=True,
    )

    img_num = -500
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        mask = fgbg.apply(frame)
        mask = dilate(mask, kernel_size=3, iterations=1)
        mask = erode(mask, kernel_size=7, iterations=1)
        mask = dilate(mask, kernel_size=2, iterations=1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        rgba[:, :, 3] = mask

        out_name = os.path.join(out, f"frame{img_num:06d}.png")
        if img_num >= 0:
            cv2.imwrite(out_name, rgba)
            print(f"Saved {out_name}")
        else:
            print(f"Skipping {img_num}")
        img_num += 1

        # cv2.imshow("frame", rgba)
        # k = cv2.waitKey(10) & 0xFF
        # if k == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("media/PC000003.mov", "/Volumes/Bloater/tmp")
