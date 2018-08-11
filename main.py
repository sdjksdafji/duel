import cv2

# Specify the paths for the 2 files
protoFile = "./data/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "./data/pose_iter_160000.caffemodel"

threshold = 0.5

num_points = 15

# construct skeloton line pairs
pose_paris = list()
for i in range(num_points - 1):
    if i != 4 and i != 7 and i != 10 and i != 13:
        pose_paris.append((i, i + 1))
pose_paris.append((8, 14))
pose_paris.append((11, 14))
pose_paris.append((1, 14))
pose_paris.append((1, 5))
 
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def draw_skeleton_on_image(frame):
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]


    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = list()
    for i in range(num_points):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    for pair in pose_paris:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 3)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    draw_skeleton_on_image(frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
