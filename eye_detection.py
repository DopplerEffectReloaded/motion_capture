import cv2 as cv
import time

CENTER_COORDS_LIST = [(630, 10), (10, 10), (10, 471), (630, 471)] # The list of coordinate positions where object is moving
RADIUS_OBJ = 10
FONT = cv.FONT_HERSHEY_SIMPLEX

def detect_and_display(frame, face_cascade, eyes_cascade, frame_count, change_phase, phase_count):

    """
    This is a helper method for detecting face and eyes per frame
    To be used in the main function eye_face_tracker
    This function tracks the eyes per frame
    :param frame: to detect
    :param face_cascade: Face haar cascade
    :param eyes_cascade: Eyes haar cascade
    :param frame_count: The number of frames elapsed since start of video stream
    :param change_phase: A boolean value indicating whether phase has to be changed
    :param phase_count: The phase of motion the object is in
    :return: void method displays frame
    """
    # Flipping the frame and getting a normalized histogram for eye-detection
    frame = cv.flip(frame, 1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    # Detecting faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:

        # Selecting region of face for scanning eyes
        faceROI = frame_gray[y : y + h, x : x + w]

        # Detecting eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)

        # Initializing eye_coords for storing data about eyes
        eye_left = ()
        eye_right = ()

        # Flag used to toggle between left eye and right eye
        flag = True

        for x2, y2, w2, h2 in eyes:
            # Caluclating center of pupil
            eyes_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            if flag:
                eye_left = eyes_center
                flag = False
            else:
                eye_right = eyes_center
                flag = True

            # Checking if eye_left or eye_right are null, if null use old vals for calculation
            if eye_left == ():
                continue
            elif eye_right == ():
                continue
            else:
                both_eyes = (eye_left, eye_right)
            
            # Getting the centre of the pupils of both eyes
            point = tuple([int(sum(q) / len(q)) for q in zip(*both_eyes)])

            # Obtaining the starting coordinates
            if frame_count == 0:
                global start
                start = point

            # Change_phase returns true when the object moves. See eye_and_face_tracker
            if change_phase:

                # Counting the phase stage and accordingly checking if eye is moving
                if phase_count == 1:
                    global end_tl
                    end_tl = point # The eye_center coords when object moves from top left position
                    difference = (start[0] - end_tl[0], start[1] - end_tl[1])
                    

                    # Checking if the eye movement is done properly and showing success or failure message
                    if difference[0] <= 10 and (difference[1] <= 5 and difference[1] >= -35):
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "SUCCESS", (200, 266), FONT, 2, (0, 255, 0), 2, cv.LINE_AA)
                        time.sleep(2)
                        print(difference)
                    else:
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "FAILURE", (200, 266), FONT, 2, (0, 0, 255), 2, cv.LINE_AA)
                        time.sleep(2)

                if phase_count == 2:
                    global end_tr
                    end_tr = point # The eye_center coords when object moves from top right position
                    difference = (end_tr[0] - end_tl[0], end_tr[1] - end_tl[1])
                    

                    # Checking if the eye movement is done properly and showing success or failure message
                    if difference[0] <= 10 and (difference[1] <= 10 and difference[1] >= -10):
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "SUCCESS", (200, 266), FONT, 2, (0, 255, 0), 2, cv.LINE_AA)
                        time.sleep(2)
                        print(difference)
                    else:
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "FAILURE", (200, 266), FONT, 2, (0, 0, 255), 2, cv.LINE_AA)
                        time.sleep(2)

                if phase_count == 3:
                    global end_br
                    end_br = point # The eye_center coords when object moves from bottom right position
                    difference = (end_br[0] - end_tr[0], end_br[1] - end_tr[1])
                    

                    # Checking if the eye movement is done properly and showing success or failure message
                    if (difference[0] <= 25 and difference[0] >= -25) and difference[1] >= 25:
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "SUCCESS", (200, 266), FONT, 2, (0, 255, 0), 2, cv.LINE_AA)
                        time.sleep(2)
                        print(difference)
                    else:
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "FAILURE", (200, 266), FONT, 2, (0, 0, 255), 2, cv.LINE_AA)
                        time.sleep(2)

                if phase_count == 4:
                    global end_bl
                    end_bl = point # The eye_center coords when object moves from bottom left position
                    difference = (end_br[0] - end_bl[0], end_br[1] - end_bl[1])
                    

                    # Checking if the eye movement is done properly and showing success or failure message
                    if difference[0] >= 40 and (difference[1] < 15 and difference[1] > -15):
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "SUCCESS", (200, 266), FONT, 2, (0, 255, 0), 2, cv.LINE_AA)
                        time.sleep(2)
                        print(difference)
                    else:
                        cv.rectangle(frame, (0, 0), (1000, 512), (255, 255, 255), -1)
                        cv.putText(frame, "FAILURE", (200, 266), FONT, 2, (0, 0, 255), 2, cv.LINE_AA)
                        time.sleep(2)

                    phase_count = 0 # setting the phase count back to zero, thus providing loop functionality

            frame = cv.circle(frame, point, radius=3, color=(0, 0, 255), thickness=-1)

    # Forcing the window to run in full screen
    cv.namedWindow("Capture-Face Detection", cv.WND_PROP_FULLSCREEN) 
    cv.setWindowProperty("Capture-Face Detection", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    #Displaying the frame
    cv.imshow("Capture-Face Detection", frame)


def eye_and_face_tracker():
    """
    This is a caller method which calls the above function for each function frame
    
    :return: void method
    """
    # Getting the haar cascade algorithm for running from local storage and loading it
    face_cascade_name = "C:\\Users\\SACHIN\\anaconda3\\Lib\\site-packages\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml"  # Path
    eyes_cascade_name = "C:\\Users\\SACHIN\\anaconda3\\Lib\\site-packages\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml"  # Path
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print("--(!)Error loading face cascade")
        exit(0) # Exits if cannot find model
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print("--(!)Error loading eyes cascade")
        exit(0) # Exits if cannot find model

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter("output_eye.avi", fourcc, 7, (640, 480))

    # Starts the video capture through camera
    cap = cv.VideoCapture(0)

    # If video capture cannot be opened this executes
    if not cap.isOpened():
        print("--(!)Error opening camera")
        exit(0)
    
    # Initializing variables to pass into detect_and_display function
    frame_count = -1
    change_phase = False
    phase_count = 0

    # Steps into the loop for displaying frames
    while True:
        ret, frame = cap.read()

        # If frame or return value is not obtained breaks the loop
        if ret is None:
            break

        frame_count += 1 # Counts the number of frames

        # Passes the change_phase signal when frame_count reaches certain numbers and increments the phase_stage
        if frame_count == 50:
            change_phase = True
            phase_count += 1
        if frame_count == 100:
            change_phase = True
            phase_count += 1
        if frame_count == 150:
            change_phase = True
            phase_count += 1
        if frame_count == 200:
            phase_count += 1

        # Checks if frame count is within limits of list
        if frame_count // 50 <= 3:
            # Creates the circular object to follow
            frame = cv.circle(frame, CENTER_COORDS_LIST[frame_count // 50], RADIUS_OBJ, color=(0, 255, 255),thickness=-1)
        else:
            # Resets the frame_count and change_phase to send it into a loop
            frame_count = 1
            change_phase = True

        # Calls the function frame-by-frame
        detect_and_display(frame, face_cascade, eyes_cascade, frame_count, change_phase, phase_count)

        # Resets change_phase so as to not continuously change the phase
        change_phase = False

        out.write(frame)

        # Waits for 'space' key to be pressed to terminate program
        if cv.waitKey(50) == ord(" "):
            break


# Calls the eye_and_face_tracker function
if __name__ == "__main__":
    eye_and_face_tracker()
