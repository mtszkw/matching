import cv2
import numpy as np


OBJECT_IMG      = "data/notebook/object_3.jpg"
ORIGINAL_IMG    = "data/notebook/original_2.jpg"


def show_good_features(src_img):
    img = src_img.copy()
    
    img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    corners = np.int0(cv2.goodFeaturesToTrack(img_gray, 25, 0.01, 10))

    for c in corners:
        x, y = c.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    cv2.imshow('Good features', img)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def match_images_ORB(img1, img2):
    orb = cv2.ORB_create()
    keypts1, descr1 = orb.detectAndCompute(img1, None)
    keypts2, descr2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    matches = bf.match(descr1, descr2)
    matches = sorted(matches, key = lambda x: x.distance)

    no_best_matches = int(0.10 * len(matches))
    print("{} matches found, selected {} best matches".format(len(matches), no_best_matches))
    out_img = cv2.drawMatches(img1, keypts1, img2, keypts2, matches[:no_best_matches], None, flags=2)

    cv2.imshow('Feature matching (ORB)', out_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def match_images_SIFT(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    keypts1, descr1 = sift.detectAndCompute(img1, None)
    keypts2, descr2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descr1, descr2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    out_img = cv2.drawMatchesKnn(img1, keypts1, img2, keypts2, good_matches, None, flags=2)
    cv2.imshow('Feature matching (SIFT)', out_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    object_img = cv2.imread(OBJECT_IMG)
    original_img = cv2.imread(ORIGINAL_IMG)
    
    #show_good_features(object_img)
    #show_good_features(original_img)

    match_images_ORB(object_img, original_img)
    match_images_SIFT(object_img, original_img)
