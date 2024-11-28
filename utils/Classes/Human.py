import time


object_ID_mapping = {}


class Human:
    '''
    All functions necessary to detect a human
    '''


    def __init__(self):
        '''
        Initialise class
        '''
        self.initially_detected = time.time()


    def detected(self):
        '''
        Mesure detection time period

        Return:
            Red: if time period < 2 seconds
            Orange: if time period is between 2-5 seconds
            Green: if time period > 5 seconds
        '''

        # Calculate detection period
        current_time = time.time() 
        DetectionPeriod = current_time - self.initially_detected

        # Return detection period along with its relavent colour 
        if DetectionPeriod < 2:
            return (0, 0, 255), DetectionPeriod
        elif 2 <= DetectionPeriod < 5:
            return (0, 165, 255), DetectionPeriod
        else:
            return (0, 255, 0), DetectionPeriod