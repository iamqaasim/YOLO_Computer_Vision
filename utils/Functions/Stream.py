import cv2
from ultralytics import YOLO
from utils.Classes.Human import Human, object_ID_mapping


def process_stream(stream_1, stream_2, yolo_model_1, yolo_model_2):
    """
    Processes two video streams, applies YOLO object detection and tracking,
    annotates the frames, and displays them in separate windows.

    Args:
        stream_1 (cv2.VideoCapture): The first video stream.
        stream_2 (cv2.VideoCapture): The second video stream.
        yolo_model (object): The YOLO model used for object detection and tracking.

    Return:
        A window displaying the annotated frames of both streams.
    """

    # Open video streams
    while stream_1.isOpened() and stream_2.isOpened():

        # Read window display data
        read_frame_bool_1, window_display_data_1 = stream_1.read()
        read_frame_bool_2, window_display_data_2 = stream_2.read()

        # Exit loop if there's a problem reading the streams
        if not read_frame_bool_1 or not read_frame_bool_2:
            break  
        
        # Resize display window to 640x480
        window_display_data_1 = cv2.resize(window_display_data_1, (640, 480))
        window_display_data_2 = cv2.resize(window_display_data_2, (640, 480))

        # Perform object tracking using YOLO
        result_1 = yolo_model_1.track(window_display_data_1, persist=True)[0]
        result_2 = yolo_model_2.track(window_display_data_2, persist=True)[0]
        
        # Yolo tracker box object data for boundry box outputs
        yolo_box_tracker_data_1 = result_1.boxes  
        yolo_box_tracker_data_2 = result_2.boxes  


        # Annotate the boundry box with tracking results
        annotate_frame(window_display_data_1, yolo_box_tracker_data_1)
        annotate_frame(window_display_data_2, yolo_box_tracker_data_2)

        # Display annotation for displayed window
        cv2.imshow("YOLO Tracking Stream 1", window_display_data_1)
        cv2.imshow("YOLO Tracking Stream 2", window_display_data_2)

        # Exit on pressing 'q'
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break

    # Release resources
    stream_1.release()
    stream_2.release()
    cv2.destroyAllWindows()


def extract_box_data(boxes):
    '''
    Extracts IDs, class IDs, confidences, and bounding boxes from the given boxes object.

    Args:
        boxes (Boxes): Boxes object containing tracking results.

    Returns:
        An array containing IDs, class IDs, confidences, and bounding boxes
    '''

    ids = boxes.id.cpu().numpy().astype(int)
    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy() 
    bboxes = boxes.xyxy.cpu().numpy().astype(int)

    return ids, class_ids, confidences, bboxes



def annotate_frame(frame, boxes):
    '''
    Annotate the frame by extracting tracking information from the boxes, checks for human objects,
    and draws bounding boxes and labels on the frame.

    Args:
        frame: The image/frame to be annotated.
        boxes: A data structure containing tracking results, which includes IDs, class IDs,
             confidences, and bounding boxes.

    Return
        The annotated frame
    '''

    # extract box data
    ids , class_ids , confidences , bboxes = extract_box_data(boxes)

    for tracker_id, class_id, conf, bbox in zip(ids, class_ids, confidences, bboxes):

        # class_id = 0 identifies if a person is human
        if class_id == 0:
            
            # Fetch list of humans identified
            human_object = object_ID_mapping.get(tracker_id, None)
            
            # Ensure the object exists
            if human_object is None:
                human_object = Human()
                object_ID_mapping[tracker_id] = human_object

            # Fetch detection time and its corresponding colour
            bounding_box_color, detection_time = human_object.detected()

            # draw bounding box frame
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, 2)

            # label the bounding box
            cv2.putText(frame, f'ID:{tracker_id} | Time:{detection_time:2f} | Conf:{conf:2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bounding_box_color, 2)
