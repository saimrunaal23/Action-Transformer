import numpy as np
from numpy import array
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt


net = cv.dnn.readNetFromTensorflow("graph_opt.pb")


inWidth = 368
inHeight = 368
thr = 0.2

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                   ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                   ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                   ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                   ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

ACT_LABELS = ['standing','check-watch','cross-arms','scratch-head','sit-down','get-up','turn-around','walking','wave1','boxing',
         'kicking','pointing','pick-up','bending','hands-clapping','wave2','jogging','jumping','pjump','running']

def adjusted_points(arr):
    list_a = []
    reducedpoints = []
    scaleandcenter = []
    for frame in range(0,30):
            reorder_indices = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
            new_points = np.array([[[arr[frame][i] for i in reorder_indices]]])
            reducedpoints.append(reduce_keypoints(new_points))
            reducedpoints= np.squeeze(reducedpoints)
            list_a.append(reducedpoints)
            reducedpoints = []
#    print(np.shape(list_a))
    return list_a
    
def reduce_keypoints(arr):
        if np.shape(arr)[2] <= 15:
            print('Keypoint number has already been reduced!')
            return
        seq_list = []
        to_prune = []
        h= [0,1,2,3,4]
        rf=[15]
        lf=[16]
        
        for group in [h, rf, lf]:
            if len(group) > 1:
                to_prune.append(group[1:])
        to_prune = [item for sublist in to_prune for item in sublist]
      
        for seq in arr:
            seq[:,h[0],:] = np.true_divide(seq[:,h,:].sum(1), (seq[:,h,:] != 0).sum(1)+1e-9)
            seq[:,rf[0],:] = np.true_divide(seq[:,rf,:].sum(1), (seq[:,rf,:] != 0).sum(1)+1e-9)
            seq[:,lf[0],:] = np.true_divide(seq[:,lf,:].sum(1), (seq[:,lf,:] != 0).sum(1)+1e-9)
            seq_list.append(seq)
        arr = np.stack(seq_list)
        arr = np.delete(arr, to_prune, 2)
        return arr

def scale_and_center(arr):
        arr = [arr]
        for X in [arr]:
            seq_list = []
            for seq in X:
                pose_list = []
                for pose in seq:
                    zero_point = (pose[1, :2] + pose[2,:2]) / 2
                    module_keypoint = (pose[7, :2] + pose[8,:2]) / 2
                    scale_mag = np.linalg.norm(zero_point - module_keypoint)
                    if scale_mag < 1:
                        scale_mag = 1
                    pose[:,:2] = (pose[:,:2] - zero_point) / scale_mag
                    pose_list.append(pose)
                seq = np.stack(pose_list)
                seq_list.append(seq)
            X = np.stack(seq_list)
        
        arr = np.delete(arr,[], 2)
        arr = np.squeeze(arr)
        # print('\n *** scale and center *** \n')
        # print(arr.shape)
        return arr

def add_velocities(seq, T=30, C=3):
    v1 = np.zeros((T+1, seq.shape[1], C-1))
    v2 = np.zeros((T+1, seq.shape[1], C-1))
    v1[1:,...] = seq[:,:,:2]
    v2[:T,...] = seq[:,:,:2]
    vel = (v2-v1)[:-1,...]
    data = np.concatenate((seq[:,:,:2], vel), axis=-1)    
    return data

if __name__ == "__main__":
    sess = rt.InferenceSession("AcT_base_model_posenet.onnx")
    input_name = sess.get_inputs()[0].name
    cap = cv.VideoCapture('clap.mp4')
    cap.set(cv.CAP_PROP_FPS, 30)
    cap.set(3, 800)
    cap.set(4,800)
    bp=[]
    list_30f = []
    count = 0
    if not cap.isOpened():
        cap = cv.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while cv.waitKey(1)<0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  

        assert(len(BODY_PARTS) == out.shape[1])

        points = []

        for i in range(len(BODY_PARTS)):

            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append([int(x), int(y)] if conf > thr else [0,0])

        if(len(bp) == 30):
            bp.pop(0)
            bp.append(points)
            adjustedpoints = adjusted_points(bp)
            scaleandcenter = scale_and_center(adjustedpoints)
            addvel = add_velocities(scaleandcenter)
            addvel = addvel.reshape(1, 30,-1)
            pred_onx = sess.run(None, {input_name: addvel.astype(np.float32)})[0]
            prediction = ACT_LABELS[np.argmax(pred_onx)]
            print(prediction)

        else:
            bp.append(points)


        if(cap.get(cv.CAP_PROP_POS_FRAMES)== 30.0):
            adjustedpoints = adjusted_points(bp)
            scaleandcenter = scale_and_center(adjustedpoints)
            addvel = add_velocities(scaleandcenter)
            addvel = addvel.reshape(1, 30,-1)
            addvel = np.squeeze(addvel)
            list_30f.append(addvel)
            list_30f = list(list_30f)
            print("When it hits 30 frames",np.shape(list_30f))

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom]!= [0,0] and points[idTo] != [0,0]:
                cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


        t, _ = net.getPerfProfile()
        freq = cv.getTickFrequency() / 1000
        cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("PoSE ESTIMATION", frame)
        key = cv.waitKey(25)
        if key == ord('n') or key == ord('p'):
            break

