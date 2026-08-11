#! /usr/bin/env python
# -*- coding: utf-8 -*-
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class LandmarksDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path="S:/SilentVoice/data/face_landmarker.task"
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        # Equivalent landmark indices from 478-point mesh
        # matching original FaceKeyPoint 0,1,2,3 (right eye, left eye, nose, mouth)
        self.KEYPOINT_INDICES = [33, 263, 1, 13]

    def __call__(self, video_frames):
        landmarks = self.detect(video_frames)
        assert any(l is not None for l in landmarks), \
            "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames):
        landmarks = []
        for frame in video_frames:
            ih, iw = frame.shape[:2]
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )
            result = self.detector.detect(mp_image)

            if not result.face_landmarks:
                landmarks.append(None)
                continue

            face = result.face_landmarks[0]  # first detected face
            lmx = []
            for idx in self.KEYPOINT_INDICES:
                lm = face[idx]
                lmx.append([int(lm.x * iw), int(lm.y * ih)])

            landmarks.append(np.array(lmx))
        return landmarks

    def close(self):
        try:
            self.detector.close()
        except Exception:
            pass