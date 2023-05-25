import cv2
import os
import os.path as osp
import numpy as np

# Color palette for lane visualization
def getcolor(code):
    if code == 1:
        return (0, 255, 0)
    if code == 2:
        return (0, 0, 255)
    if code == 3:
        return (255, 255, 0)
    if code == 4:
        return (0, 255, 255)
    if code == 5:
        return (255, 0, 255)
    if code == 6:
        return (45, 88, 200)
    if code == 7:
        return (213, 22, 224)


def imshow_lanes(img, lanes, show=False, out_file=None, lane_classes = None, num_classes=2):
    img = np.zeros((720, 1280, 3))
    
    if lane_classes is not None:
        if num_classes == 6:
            cv2.putText(img,'solid-yellow',(0,40), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(1),2,cv2.LINE_AA)
            cv2.putText(img,'solid-white',(0,70), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(2),2,cv2.LINE_AA)
            cv2.putText(img,'dashed',(0,100), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(3),2,cv2.LINE_AA)
            cv2.putText(img,'Botts\'-dots',(0,170), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(4),2,cv2.LINE_AA)
            cv2.putText(img,'double-solid-yellow',(0,200), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(5),2,cv2.LINE_AA)
            cv2.putText(img,'unknown',(0,230), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(6),2,cv2.LINE_AA)
        else:
            cv2.putText(img,'solid',(0,40), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(1),2,cv2.LINE_AA)
            cv2.putText(img,'dashed',(0,70), cv2.FONT_HERSHEY_SIMPLEX, 1,getcolor(2),2,cv2.LINE_AA)
            #df = {0:0, 1:1, 2:1, 3:2, 4:2, 5:1, 6:1}
            #lane_classes = list(map(df.get,lane_classes))


    for i, lane in enumerate(lanes):
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            if lane_classes is not None:
                color = getcolor(lane_classes[i])
            else:
                color = (255, 0, 0)
            cv2.circle(img, (x, y), 4, color, 2)
    
    '''
    for i, lane in enumerate(lanes):
        for j in range(len(lane)-1):
            if lane_classes is not None:
                color = getcolor(lane_classes[i])
            else:
                color = (255, 0, 0)
            cv2.line(img, lane[j], lane[j+1], color, 5)
    '''

    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if out_file:
        if not osp.exists(osp.dirname(out_file)):
            os.makedirs(osp.dirname(out_file))
        cv2.imwrite(out_file, img)

