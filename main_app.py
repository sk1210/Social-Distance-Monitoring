import cv2
import sys
import numpy as np
from utils import transparentOverlay, dst_circle, get_bounding_box, int_circle

# FLAGS
WRITE_VIDEO = True
SHOW_OUTPUT = True


def main(video_source):
    # load yolov2
    net = cv2.dnn.readNet("yolov2.weights", "yolov2.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # input capture object
    cap = cv2.VideoCapture(video_source)

    if WRITE_VIDEO:
        # output video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if not (i % 3 == 0): continue
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_id = get_bounding_box(outs, height, width)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(class_id)
        #
        circles = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]

                # draw circle
                cx, cy = x + w // 2, y + h
                frame = cv2.line(frame, (cx, cy), (cx, cy - h // 2), (0, 255, 0), 2)
                frame = cv2.circle(frame, (cx, cy - h // 2), 5, (255, 20, 200), -1)
                circles.append([cx, cy - h // 2, h])

        int_circles_list = []
        indexes = []
        for i in range(len(circles)):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, len(circles)):
                x2, y2, r2 = circles[j]
                if int_circle(x1, y1, x2, y2, r1 // 2, r2 // 2) >= 0 and abs(y1 - y2) < r1 // 4:
                    indexes.append(i)
                    indexes.append(j)

                    int_circles_list.append([x1, y1, r1])
                    int_circles_list.append([x2, y2, r2])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            rows, cols, _ = frame.shape

            for i in range(len(circles)):
                x, y, r = circles[i]

                if i in indexes:
                    color = (0, 0, 255)
                else:
                    color = (0, 200, 20)
                scale = (r) / 100
                transparentOverlay1(frame, dst_circle, (x, y - 5), alphaVal=110, color=color, scale=scale)

            cv2.rectangle(frame, (0, rows - 80), (cols, rows), (0, 0, 0), -1)
            cv2.putText(frame,
                        "Total Persons : " + str(len(boxes)),
                        (20, rows - 40),
                        fontFace=cv2.QT_FONT_NORMAL,
                        fontScale=1,
                        color=(215, 220, 245))

            cv2.putText(frame,
                        "Defaulters : " + str(len(set(indexes))),
                        (cols - 300, rows - 40),
                        fontFace=cv2.QT_FONT_NORMAL,
                        fontScale=1,
                        color=(0, 220, 245))
        if WRITE_VIDEO:
            out.write(frame)

        if SHOW_OUTPUT:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if WRITE_VIDEO:
        out.release()

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    # path to input video
    video = "input2.mp4"
    if len(sys.argv) > 1:
        video_source = sys.arg[1]
    else:
        video_source = "input2.mp4"

    main(video_source=video)
