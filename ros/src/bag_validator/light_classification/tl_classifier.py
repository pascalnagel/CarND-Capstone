from styx_msgs.msg import TrafficLight
from PIL import Image
from PIL.ImageDraw import Draw
from PIL import ImageFont
import numpy as np
import tensorflow as tf
from collections import Counter

class TLClassifier(object):
    def __init__(self):
        self.frozen_graph_filename = "/capstone/ros/src/tl_detector/light_classification/models/final_model/frozen_inference_graph.pb"
        self.graph = self.load_graph(self.frozen_graph_filename)
        self.sess = tf.Session(graph=self.graph)
        self.prediction_counter = 0
        self.class_map_str = {1: 'RED',
                              2: 'YELLOW',
                              3: 'GREEN'}
        self.class_map = {1: TrafficLight.RED,
                          2: TrafficLight.GREEN,
                          3: TrafficLight.YELLOW}
        self.save_pred_images = True
        self.font = ImageFont.truetype('light_classification/arial.ttf', 22) #font size for labels

    def load_graph(self, frozen_graph_filename):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
        return graph

    def get_classification(self, inp_image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_array = np.array(inp_image)
        image = np.reshape(image_array, (1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        results = self.sess.run([self.graph.get_tensor_by_name('prefix/detection_boxes:0'), 
                                self.graph.get_tensor_by_name('prefix/detection_scores:0'), 
                                self.graph.get_tensor_by_name('prefix/detection_classes:0')], 
                                {self.graph.get_tensor_by_name('prefix/image_tensor:0'): image})

        pilimg = Image.fromarray(image_array)
        classes = []
        for boxid in range(len(results[0][0])):
            if results[1][0][boxid] > 0.5:
                predclass = results[2][0][boxid]
                classes.append(predclass)
                if self.save_pred_images:
                    width = pilimg.size[0]
                    height = pilimg.size[1]
                    box = results[0][0][boxid]
                    xmin = int(box[1]*width)
                    ymin = int(box[0]*height)
                    xmax = int(box[3]*width)
                    ymax = int(box[2]*height)
                    draw = Draw(pilimg)
                    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red')
                    draw.text((xmax+5, ymin+5), "Class: "+str(self.class_map_str[predclass]), font=self.font)
        
        if self.save_pred_images:
            pilimg.save("/capstone/ros/classifier_output/bag/{:0>5d}.jpg".format(self.prediction_counter))
            self.prediction_counter += 1

        if classes:
            keys = Counter(classes).keys()
            vals = Counter(classes).values()
            pred_class = keys[np.argmax(vals)]
            return self.class_map[pred_class]
        else:
            return TrafficLight.UNKNOWN