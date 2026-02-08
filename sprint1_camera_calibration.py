"""
Sprint 1: Camera Calibration Module
Road Defect Detection System
Team: Naa Lamle Boye, Thomas Kojo Quarshie, Chelsea Owusu, Elijah Boateng
"""

import numpy as np
import cv2
import os


def calibrate_camera(video_files, checkerboard_size=(8, 5), sample_rate=1, output_file="camera_calib.npz"):
    """
    Calibrate camera using checkerboard videos.
    
    Args:
        video_files: List of video file paths
        checkerboard_size: Tuple of (width, height) internal corners
        sample_rate: Process every Nth frame (1 = all frames)
        output_file: Output file path for calibration data
    
    Returns:
        dict: Calibration results with keys: mtx, dist, rvecs, tvecs, rms_error
    """
    CHECKERBOARD = checkerboard_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
    CORNER_REFINE_WINDOW = (15, 15)
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    image_size = None
    
    print(f"Processing {len(video_files)} video(s)...")
    
    for video_path in video_files:
        if not os.path.exists(video_path):
            print(f"Warning: {video_path} not found, skipping")
            continue
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}, skipping")
            continue
        
        frame_count = 0
        frames_found = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if image_size is None:
                    image_size = gray.shape[::-1]
                
                # Try multiple detection strategies
                ret_corners, corners = cv2.findChessboardCorners(
                    gray, CHECKERBOARD,
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
                    cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS
                )
                
                if not ret_corners:
                    ret_corners, corners = cv2.findChessboardCorners(
                        gray, CHECKERBOARD,
                        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
                        cv2.CALIB_CB_FAST_CHECK
                    )
                
                if not ret_corners:
                    ret_corners, corners = cv2.findChessboardCorners(
                        gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH
                    )
                
                if ret_corners:
                    corners2 = cv2.cornerSubPix(
                        gray, corners, CORNER_REFINE_WINDOW, (-1, -1), criteria
                    )
                    
                    corner_coords = corners2.reshape(-1, 2)
                    x_range = np.max(corner_coords[:, 0]) - np.min(corner_coords[:, 0])
                    y_range = np.max(corner_coords[:, 1]) - np.min(corner_coords[:, 1])
                    
                    min_coverage = 0.08
                    if x_range > image_size[0] * min_coverage and y_range > image_size[1] * min_coverage:
                        objpoints.append(objp)
                        imgpoints.append(corners2)
                        frames_found += 1
            
            frame_count += 1
        
        cap.release()
        print(f"  {os.path.basename(video_path)}: {frames_found} frames")
    
    if len(objpoints) < 3:
        raise ValueError(f"Insufficient frames for calibration: {len(objpoints)}")
    
    print(f"\nCalibrating with {len(objpoints)} frames...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=0
    )
    
    np.savez(output_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"RMS Error: {ret:.4f} pixels")
    print(f"Calibration saved to {output_file}")
    
    return {
        'mtx': mtx,
        'dist': dist,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'rms_error': ret,
        'num_frames': len(objpoints)
    }


if __name__ == "__main__":
    video_files = [
        'checkerboard/IMG_1059.MOV',
        'checkerboard/IMG_1060.MOV',
        'checkerboard/IMG_1061.MOV',
        'checkerboard/IMG_1062.MOV'
    ]
    
    result = calibrate_camera(video_files)
    print(f"\nCalibration complete. RMS Error: {result['rms_error']:.4f} pixels")
