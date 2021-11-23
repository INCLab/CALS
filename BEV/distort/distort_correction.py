import numpy as np
import cv2
import glob
import argparse

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=6, height=4):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # Check chessboard is detected
        print(ret)
        print(corners)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

            cv2.imshow('win', img)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=cv2.CALIB_FIX_K2)

    return [ret, mtx, dist, rvecs, tvecs]


def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


def undistort(img_path, save_path, prefix, image_format, mtx, dist):
    images = glob.glob(img_path + '/' + prefix + '*.' + image_format)

    for fname in images:
        sname = fname.split('\\')[-1]
        print(sname)
        img = cv2.imread(fname)

        h, w = img.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        file_path = save_path + '/' + sname

        cv2.imwrite(file_path, dst)


if __name__ == '__main__':
    image_dir = './chess_img'
    prefix = 'chess'
    image_format = 'jpg'
    square_size = 30  # 2cm
    save_file = './saved_coeffi/coeffi.yaml'

    ret, mtx, dist, rvecs, tvecs = calibrate(image_dir, prefix, image_format, square_size)

    save_coefficients(mtx, dist, save_file)
    print("Calibration is finished. RMS: ", ret)


    save_dir = './test_img'
    undistort(image_dir, save_dir, prefix, image_format, mtx, dist)
    print('Save correct distortion images.')