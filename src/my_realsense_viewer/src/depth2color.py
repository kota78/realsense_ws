#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class DepthImageConverter:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)
    
    def depth_callback(self, data):
        try:
            # 深度画像をROSメッセージからOpenCV形式に変換
            depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
            # 深度画像を可視化のために8ビットにスケール変換
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image_8bit = cv2.convertScaleAbs(depth_image_normalized)

            # HSVカラースペースに変換してカラー表示を実現
            rows, cols = depth_image_8bit.shape
            img_max = 255 * np.ones([rows, cols], dtype=np.uint8)

            img_h = depth_image_8bit
            img_s = img_max
            img_v = depth_image_8bit

            img_hsv = cv2.merge((img_h, img_s, img_v))
            img_dst = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow("Depth Image Window", img_dst)
            cv2.waitKey(3)
        except CvBridgeError as e:
            rospy.logerr(e)

def main():
    rospy.init_node('depth_image_converter', anonymous=True)
    dic = DepthImageConverter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
