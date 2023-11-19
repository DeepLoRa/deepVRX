import cv2
import threading
import queue
from rknnlite.api import RKNNLite
import numpy as np
import time

# 自定义无缓存读视频类
class VideoCapture:
    def __init__(self, name):
        frameWidth = 480
        frameHeight = 320
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, frameWidth)
        self.cap.set(4, frameHeight)
        self.cap.set(6,1196444237)      # 设置格式为MJPG模式
        self.cap.set(10, 50)            # 设置亮度
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,3)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False    # to gracefully close sub-thread
        th = threading.Thread(target=self._reader)
        th.daemon = True             # 设置工作线程为后台运行
        th.start()
        
    # 实时读帧，只保存最后一帧
    def _reader(self):
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() 
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def terminate(self):
        self.stop_threads = True
        self.cap.release()

rknn_model = 'deepVRX_rk3588s.rknn'

#----------init rknn_lite2-----------
rknn_lite = RKNNLite()
print('--> Load RKNN model')
ret = rknn_lite.load_rknn(rknn_model)
if ret != 0:
    print('Load RKNN model failed')
print('Load RKNN model failed done')
print('--> Init runtime environment')
ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
if ret != 0:
    print('Init runtime environment failed')
print('Init runtime environment done')

cap = VideoCapture(0)
fpsLimit = 60
while type(cap.read()) == None:
    pass

while True:
    fpsTimer = time.time()
    ret,frame = cap.read()
    frame = cv2.resize(frame,(480,360))
    output = rknn_lite.inference(inputs=[frame])[0][0]
    
    output = output.transpose((1, 2, 0))  #转换维度
    output = output*0.5+0.5  #反归一化
    output = (output * 255).astype(np.uint8)  #转换为整数形式
    output = cv2.cvtColor(output,cv2.COLOR_RGB2BGR)
    
    cv2.imshow('test output', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下“q”退出
        print("Quit")
        break
    
    if 1/(time.time()-fpsTimer) >= fpsLimit:
        time.sleep(max(0,1/fpsLimit-(time.time()-fpsTimer)))
    fps = 1/(time.time() - fpsTimer)
