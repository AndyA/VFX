import cv2
import numpy as np


def dilate(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


def erode(img: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


def main(src: str):
    cap = cv2.VideoCapture(src)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=5000,
        varThreshold=16,
        detectShadows=True,
    )

    img_num = -500
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        fgmask = fgbg.apply(frame)
        mask1 = dilate(fgmask, kernel_size=3, iterations=1)
        mask2 = erode(mask1, kernel_size=5, iterations=1)
        mask3 = dilate(mask2, kernel_size=5, iterations=1)
        mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
        rgba[:, :, 3] = mask4
        if img_num >= 0:
            cv2.imwrite(f"tmp/frame{img_num:06d}.png", rgba)
            print(f"Saved frame{img_num:06d}.png")
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
    main("media/PC000003.mov")
