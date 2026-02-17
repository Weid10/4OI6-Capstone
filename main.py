import cv2 as cv
import capture
import yolo_picam_sparecup

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
    picam2 = capture.camera_init()
    yolo_picam_sparecup.main(picam2)
    picam2.stop()