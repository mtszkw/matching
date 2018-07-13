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


def find_object(object_img, original_img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypts1, descr1 = sift.detectAndCompute(object_img, None)
    keypts2, descr2 = sift.detectAndCompute(original_img, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descr1, descr2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])

    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ keypts1[m[0].queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([ keypts2[m[0].trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h, w, d = object_img.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(original_img, [np.int32(dst)],True,(100,250,200),3, cv2.LINE_AA)

    else:
        matchesMask = None

    out_img = cv2.drawMatchesKnn(object_img, keypts1, original_img, keypts2, good_matches, None, flags=2)

    cv2.imshow('Feature matching (SIFT)', out_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    object_img = cv2.imread(OBJECT_IMG)
    original_img = cv2.imread(ORIGINAL_IMG)
    
    # show_good_features(object_img)
    # show_good_features(original_img)

    # match_images_ORB(object_img, original_img)
    # match_images_SIFT(object_img, original_img)

    find_object(object_img, original_img)