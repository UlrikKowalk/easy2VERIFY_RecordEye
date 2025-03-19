
import mediapipe as mp
import numpy as np
import cv2 as cv

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_CENTER_LINE = [8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164]
LEFT_EYE_BORDERS = [362, 263]
RIGHT_EYE_BORDERS = [33, 133]
EYELID_LEFT_TOP = [160, 159, 158]
EYELID_LEFT_BOTTOM = [144, 145, 153]
EYELID_RIGHT_TOP = [385, 386, 387]
EYELID_RIGHT_BOTTOM = [380, 374, 373]


class FaceMapping:

    def __init__(self):

        self.results = None
        self.center_left = None
        self.center_right = None
        self.face_x = None
        self.face_y = None
        self.gaze_x = None
        self.gaze_y = None
        self.azimuth = 0
        self.elevation = 0
        self.tilt = 0
        self.is_blink = False
        self.eye_boundaries_left = None
        self.eye_boundaries_right = None
        # self.userHead = 3400
        # self.distance = 45
        self.cameraOffset = 0#10#4.75
        self.camhfov = 40#47.0#50.0 #-> GOPRO/4-3Linear, #70.4 -> Logitech 920,  #47.0 -> MacbookPro # camera horizontal field of view, DEGREES
        self.camvfov = 40#26.5#38.7 #-> GOPRO/4-3Linear #43.4 -> Logitech 920, #26.5 -> MacbookPro  # camera vertical field
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                               refine_landmarks=True,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)

    def first(self, frame):

        self.img_h, self.img_w = frame.shape[:2]

        # The camera matrix
        self.focal_length = 1 * self.img_w

        self.cam_matrix = np.array([[self.focal_length, 0, self.img_h / 2],
                                    [0, self.focal_length, self.img_w / 2],
                                    [0, 0, 1]])

        # The Distance Matrix
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)

    def calculate_face_orientation(self, frame):

        self.results = self.face_mesh.process(frame)
        if self.results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [self.img_w, self.img_h]).astype(int) for p in
                 self.results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            self.center_left = np.array([l_cx, l_cy], dtype=np.int32)
            self.center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # print(np.sum(abs(mesh_points[EYELID_LEFT_TOP] - mesh_points[EYELID_LEFT_BOTTOM]), axis=(0, 1)))

            # if (abs(mesh_points[EYELID_LEFT_TOP][1] - mesh_points[EYELID_LEFT_BOTTOM][1]) < 1 and
            #     abs(mesh_points[EYELID_LEFT_TOP][0] - mesh_points[EYELID_LEFT_BOTTOM][0]) < 1) or \
            #         (abs(mesh_points[EYELID_RIGHT_TOP][1] - mesh_points[EYELID_RIGHT_BOTTOM][1]) < 1 and
            #          abs(mesh_points[EYELID_RIGHT_TOP][0] - mesh_points[EYELID_RIGHT_BOTTOM][0]) < 1):
            #
            if (np.sum(abs(mesh_points[EYELID_LEFT_TOP] - mesh_points[EYELID_LEFT_BOTTOM]), axis=(0, 1)) < 10) or \
                    (np.sum(abs(mesh_points[EYELID_RIGHT_TOP] - mesh_points[EYELID_RIGHT_BOTTOM]), axis=(0, 1)) < 10):
                self.is_blink = True
            else:
                self.is_blink = False

            # if self.center_left.all() and self.center_right.all():
                # paint circles on irises
                # cv.circle(frame, self.center_left, int(l_radius), (0, 255, 0), 1, cv.LINE_AA)
                # cv.circle(frame, self.center_right, int(r_radius), (0, 255, 0), 1, cv.LINE_AA)
                # frame = cv.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

            nose_coordinates = np.array(mesh_points[NOSE_CENTER_LINE])
            # Obtain face center from nose
            if nose_coordinates.all():
                self.face_x, self.face_y = np.mean(nose_coordinates, axis=0)

            self.eye_boundaries_left = np.array(mesh_points[LEFT_EYE_BORDERS])
            self.eye_boundaries_right = np.array(mesh_points[RIGHT_EYE_BORDERS])

            # left_iris = mesh_points[LEFT_IRIS]
            # right_iris = mesh_points[RIGHT_IRIS]
            # irl = np.mean(left_iris,axis=0)
            # dist1 = np.sqrt((irl[0]-self.eye_boundaries_left[0,0])**2 + (irl[1]-self.eye_boundaries_left[0,1])**2)

            if self.eye_boundaries_left.all() and self.eye_boundaries_right.all():
                # paint white line on eye center
                # cv.line(frame, self.eye_boundaries_left[0], self.eye_boundaries_left[1], (255, 255, 255), thickness=1)
                # cv.line(frame, self.eye_boundaries_right[0], self.eye_boundaries_right[1], (255, 255, 255), thickness=1)

                self.gaze_x = np.mean([np.mean(self.eye_boundaries_left, axis=0)[0] - self.center_left[0],
                                       np.mean(self.eye_boundaries_right, axis=0)[0] - self.center_right[0]])
                self.gaze_y = np.mean([np.mean(self.eye_boundaries_left, axis=0)[1] - self.center_left[1],
                                       np.mean(self.eye_boundaries_right, axis=0)[1] - self.center_right[1]])

        return frame

    def get_center(self):
        return self.face_x, self.face_y

    def get_iris(self):
        return self.center_left, self.center_right, self.is_blink

    def get_eye_boundaries(self):
        return self.eye_boundaries_left, self.eye_boundaries_right

    def draw_border_around_frame(self, frame):
        return cv.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 20)

    def add_text(self, frame, text, coordinates=None):
        if coordinates is not None:
            return cv.putText(frame, text, (coordinates[0], coordinates[1]), cv.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 2, cv.LINE_AA)
        else:
            return cv.putText(frame, text, (20, 120), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2, cv.LINE_AA)

    def get_gaze(self):
        return -self.gaze_x, self.gaze_y, self.is_blink

    def get_position_data(self):

        face_3d = []
        face_2d = []

        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                        x, y = int(lm.x * self.img_w), int(lm.y * self.img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, self.cam_matrix, self.dist_matrix)

                # Get rotational matrix
                rmat, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                # Get the y rotation degree
                self.elevation = angles[0] * 360
                self.azimuth = angles[1] * 360
                self.tilt = angles[2] * 360
                return self.azimuth, self.elevation, self.tilt
