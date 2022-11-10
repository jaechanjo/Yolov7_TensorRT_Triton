import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2

import torch
import random
import torch.utils.data as data


class BaseEngine(object):
    def __init__(self, engine_path, imgsz=(640,640)):
        self.imgsz = imgsz
        self.mean = None
        self.std = None
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
        
        ####ICudaEngine(모델의 정보를 갖고 있음: 1. device_memory_size, 2.max_batch_size 3.num_bindings(I/O_bindings) 4.trt모델의 layer 개수)####
        logger = trt.Logger(trt.Logger.WARNING) 
        trt.init_libnvinfer_plugins(logger,'') #plugin 사용을 위한 초기화
        runtime = trt.Runtime(logger) #serialized ICudaEngine을 deserialize하기 위한 클래스 객체 (*serialize: 나중에 재사용하기 위한 포맷(.trt: bytestream으로 저장 됨.)으로 바꾸는 것 <-> 실행하기 위해 빌드)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine) #trt 모델을 읽어 serialized한 ICudaEngine을 deserialize 한다.
        self.context = engine.create_execution_context() #ICudaEngine을 이용해 inference를 하기 위한 context class 생성.
        ####Setup I/O binding####
        #trt 모델의 input과 output 정보를 저장하고 이후 inference 시에 사용
        self.inputs, self.outputs, self.bindings = [], [], [] 
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) #binding의 shape에 따른 volume 할당 size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype) 
            device_mem = cuda.mem_alloc(host_mem.nbytes) #해당 size에 따른 mem gpu 할당
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding): # input
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else: #output
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu (host to device)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu (device to host)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        data = []
        for out in self.outputs:
            data.append(out['host'])
        return data

    def inference(self, img_path=False, img=None, path=None, conf=0.25):
        #추론 input을 path로 받을지 혹은 이미지(nd.array)로 받을지
        if not img_path:
            origin_img = cv2.imread(path)
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        else:
            origin_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std) #preprocess
        num, final_boxes, final_scores, final_cls_inds = self.infer(img) #infer() 
        final_boxes = np.reshape(final_boxes, (-1, 4))
        num = num[0]
        if num >0:
            final_boxes, final_scores, final_cls_inds = final_boxes[:num]/ratio, final_scores[:num], final_cls_inds[:num]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)                      
        return origin_img
    
    # def batch_inference(self, batch, conf=0.25):
    #     #추론 input을 path로 받을지 혹은 이미지(nd.array)로 받을지
    #     img for img in batch
    #     img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std) #preprocess
    #     num, final_boxes, final_scores, final_cls_inds = self.infer(img) #infer() 
    #     final_boxes = np.reshape(final_boxes, (-1, 4))
    #     num = num[0]
    #     if num >0:
    #         final_boxes, final_scores, final_cls_inds = final_boxes[:num]/ratio, final_scores[:num], final_cls_inds[:num]
    #         origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
    #                          conf=conf, class_names=self.class_names)
    #     origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)                      
    #     return origin_img

    def get_fps(self):
        # warmup
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(20):
            _ = self.infer(img) #warmup
        
        #average
        t_sum = 0
        for _ in range(200):
            t_b = time.perf_counter()
            _ = self.infer(img)
            t_sum += (time.perf_counter() - t_b)
        average_t = t_sum / 200
        print(f"Average Latency: {average_t} ms")
        print(f"Average Throughput: {round(1/average_t, 1)} ips")

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1]) #640-imgsize에 맞춰주기
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32) #resize by imgsize
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None: #normalize
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id % 80] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id % 80]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id % 80] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def file_load(txt_path):
        
        data_path = []
        f = open(txt_path,'r')
        while True:
            line = f.readline()
            if not line: break
            data_path.append(line[:-1])
        f.close()
        return data_path

class CoCoDataset(data.Dataset):
    def __init__(self, txt_path):
        super(CoCoDataset, self).__init__()
        random.shuffle
#         """
#         opt_data : 'train', 'validation'
        
#         """
        self.file_list = file_load(txt_path)
        # y = pd.read_csv('audio_data/train_answer.csv', index_col=0)
        # self.y = y.values
        
    def __getitem__(self, index):
        
        x = cv2.imread(f'./coco{self.file_list[index][1:]}')
        self.x_data = torch.from_numpy(x).float()
        # self.y_data = torch.from_numpy(self.y[index]).float()
        return self.x_data #, self.y_data

    def __len__(self):
        return len(self.file_list)

def _collate_fn(batch):
    
    """
    Args:
        batch: list, len(batch) = 16.
    Returns:
        x_tensor : B, C, H, W ; tensor
        
    ex)
    torch.Size([640, 480, 3])
    torch.Size([479, 640, 3])
    """

    x_tensor=[]
    y_tensor=[]
    for img in batch:
        img = np.asarray(img)
        img = cv2.resize(img, (480, 640), cv2.INTER_CUBIC)
        x = torch.from_numpy(img)
        x_tensor.append(x)

    return x_tensor


if __name__ == "__main__":

    
    pred = BaseEngine(engine_path='./yolov7-tiny_1-fp16-nms.trt')
    origin_img = pred.inference(img_path=False, path='inference/images/horses.jpg')
    