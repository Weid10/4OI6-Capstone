import cv2 as cv
import capture

SAVE_PHOTO = False

def main():
    if SAVE_PHOTO:
        capture.take_photo("samples/saved_photo.jpg")
    else:
        array = capture.take_photo(None)

        cv.imshow("Captured Photo", array)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()