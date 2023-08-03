import cv2
import os


def main():
    __here__ = os.path.dirname(__file__)
    d0 = cv2.imread(f"{__here__}/data/forcindy/0.png")
    print(d0.shape)

    cv2.imshow("image", d0)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return


main()
