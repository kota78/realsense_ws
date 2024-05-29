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
        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_callback)
        self.depth_image = None
        self.color_image = None
        self.mask = None
        self.min_depth = 500  # 最小距離（ミリメートル）
        self.max_depth = 600 # 最大距離（ミリメートル）

    def depth_callback(self, data):
        try:
            # 深度画像をROSメッセージからOpenCV形式に変換
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")

            # カラー画像が存在する場合、深度画像をリサイズ
            if self.color_image is not None:
                self.depth_image = cv2.resize(self.depth_image, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 指定距離範囲内のマスクを作成
            self.mask = cv2.inRange(self.depth_image, self.min_depth, self.max_depth)

            # マスクを白黒で表示
            # cv2.imshow("Depth Mask", self.mask)
            # cv2.waitKey(3)
        except CvBridgeError as e:
            rospy.logerr(e)

    def color_callback(self, data):
        try:
            # RGB画像をROSメッセージからOpenCV形式に変換
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # RGB画像のサイズをデバッグ出力
            rospy.loginfo(f"Color image size: {self.color_image.shape}")

            # 深度画像が存在する場合、カラー画像のサイズにリサイズ
            if self.depth_image is not None:
                self.depth_image = cv2.resize(self.depth_image, (self.color_image.shape[1], self.color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                self.mask = cv2.inRange(self.depth_image, self.min_depth, self.max_depth)

            # マスクが作成されている場合にRGB画像に適用
            if self.mask is not None:
                # マスクを3チャンネルに変換
                mask_3ch = cv2.merge([self.mask, self.mask, self.mask])

                # サイズの確認
                rospy.loginfo(f"3-channel mask size: {mask_3ch.shape}")

                # マスクをRGB画像に適用して切り抜き
                masked_color_image = cv2.bitwise_and(self.color_image, mask_3ch)

                # 結果を表示
                cv2.imshow("Masked RGB Image", masked_color_image)
                cv2.waitKey(3)

            # オリジナルのRGB画像を表示
            # cv2.imshow("Original RGB Image", self.color_image)
            # cv2.waitKey(3)
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
