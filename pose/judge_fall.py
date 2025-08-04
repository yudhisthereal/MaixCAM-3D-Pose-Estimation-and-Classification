import os
import numpy as np

def get_fall_info(obj, online_targets_det, online_targets, index, fallParam, queue_size, fps):
    fall_down = False
    # 当前数据
    cur_bbox = [online_targets_det.x, online_targets_det.y, online_targets_det.w, online_targets_det.h]    # list
    cur_points = obj.points  
    # 历史数据
    pre_bbox = online_targets["bbox"][index].get()
    pre_points = online_targets["points"][index].get()



    # 跌倒判断  矩形框左上角纵坐标变化过快
    if (fps > 0):
        v_bbox_y = 1.0 * (pre_bbox[1] - cur_bbox[1]) / (queue_size * 1000 / fps)
    else:
        v_bbox_y = 1.0 * (pre_bbox[1] - cur_bbox[1]) / (queue_size * 1000)
    
    #print("v_bbox_y:{}".format(v_bbox_y))
    if(v_bbox_y<0 and abs(v_bbox_y)>fallParam["v_bbox_y"]):  # 跌倒
        fall_down = True

    #print("v_bbox_y:{}, fallParam: {}".format(v_bbox_y, fallParam["v_bbox_y"]))

    if not (pre_points[15*2+1] == -40 or pre_points[16*2+1] == -40 or cur_points[15*2+1] == -40 or cur_points[16*2+1] == -40 or \
       pre_points[5*2+1] == -40 or  pre_points[6*2+1] == -40 or cur_points[5*2+1] == -40 or cur_points[6*2+1] == -40):
        # 连续多帧脚跟和肩膀差不多高为跌倒
        pre_left_ankle_y = pre_points[15*2+1]
        pre_right_ankle_y = pre_points[16*2+1]
        cur_left_ankle_y = cur_points[15*2+1]
        cur_right_ankle_y = cur_points[16*2+1]
                                
        pre_left_shoulder_y = pre_points[5*2+1]
        pre_right_shoulder_y = pre_points[6*2+1]
        cur_left_shoulder_y = cur_points[5*2+1]
        cur_right_shoulder_y = cur_points[6*2+1]
        if pre_left_ankle_y<pre_left_shoulder_y and pre_right_ankle_y < pre_right_shoulder_y and cur_left_ankle_y < cur_left_shoulder_y and cur_right_ankle_y < cur_right_shoulder_y:
            fall_down = True
    
    if not (pre_points[5*2] == 0 or pre_points[6*2] == 0 or pre_points[11*2] == 0 or pre_points[12*2] == 0 or \
       cur_points[5*2] == 0 or cur_points[6*2] == 0 or cur_points[11*2] == 0 or cur_points[12*2] == 0):
        # 判断躯干角度与横坐标的夹角
        pre_mid_shoulder_x = (pre_points[5*2] + pre_points[6*2]) / 2
        pre_mid_shoulder_y = (pre_points[5*2+1] + pre_points[6*2+1]) / 2
        pre_mid_hip_x = (pre_points[11*2] + pre_points[12*2]) / 2
        pre_mid_hip_y = (pre_points[11*2+1] + pre_points[12*2+1]) / 2
        vector1 = np.array((pre_mid_shoulder_x - pre_mid_hip_x, pre_mid_shoulder_y - pre_mid_hip_y))
        vectorx = np.array((1,0))
        cur_mid_shoulder_x = (cur_points[5*2] + cur_points[6*2]) / 2
        cur_mid_shoulder_y = (cur_points[5*2+1] + cur_points[6*2+1]) / 2
        cur_mid_hip_x = (cur_points[11*2] + cur_points[12*2]) / 2
        cur_mid_hip_y = (cur_points[11*2+1] + cur_points[12*2+1]) / 2
                                
        vector1 = np.array((pre_mid_shoulder_x - pre_mid_hip_x, pre_mid_shoulder_y - pre_mid_hip_y))
        vector2 = np.array((cur_mid_shoulder_x - cur_mid_hip_x, cur_mid_shoulder_y - cur_mid_hip_y))
        vectorx = np.array((1,0))    
                                
        pre_cos_ = (vector1.dot(vectorx)) / (np.sqrt(vector1.dot(vector1)) * np.sqrt(vectorx.dot(vectorx)))
        pre_angle = np.arccos(pre_cos_) * 180 / np.pi

        cur_cos_ = (vector2.dot(vectorx)) / (np.sqrt(vector2.dot(vector2)) * np.sqrt(vectorx.dot(vectorx)))
        cur_angle = np.arccos(cur_cos_) * 180 / np.pi
        print("pre_angle:{}, cur_angle:{}".format(pre_angle, cur_angle))                                                 
        if (pre_angle < fallParam["angle"] or pre_angle > (180 - fallParam["angle"])) and (cur_angle < fallParam["angle"] or cur_angle > (180 - fallParam["angle"])):
            fall_down = True
    # # 如果w>h
    # if cur_bbox[2] > cur_bbox[3] and pre_bbox[2] > pre_bbox[3]:
    #     fall_down = False

        
    return fall_down

