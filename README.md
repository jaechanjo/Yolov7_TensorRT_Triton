# Yolov7_TensorRT_Triton
  - yolov7 pytorch 모델에 대해 Nvidia TensorRT 버전 8.0.1을 사용한 경량화 및 최적화 실험이다.
  - 21.08 Triton 서버에 yolov7 tensorrt 모델 배포했다.
  - Project로 CCTV 비정상 행동 중 실신 행동을 탐지하는 주제를 진행했다.

## Performance

**94.4 infer/sec(16 concurrency)**

![Performance](https://github.com/jaechanjo/Yolov7_TensorRT_Triton/blob/main/docs/images/Performace.png?raw=true)

> 배치 1로 16개 concurrency가, triton dynamic batch 덕분에, batch 16으로 실행하는 것과 동일하다.

## Project Introduction

  1. YOLOv7, 가장 최신 버전의 Object Detection 모델 적용
  
  
    - TAO Toolkit (current version, 4.0기준) : YOLOv4 업데이트
  2. CCTV 이상 행동 중 실신 탐지 가능
  
  
    - 기존 MSCOCO 사전학습 모델은 Standing Person만 탐지 가능
  3. 실생활 적용을 위한 실시간 구동 추론 속도 확인
  
  
    - Triton Server 기준 94.4 infer/sec(16 concurrency)

![Pipeline](https://github.com/jaechanjo/Yolov7_TensorRT_Triton/blob/main/docs/images/Pipeline.PNG?raw=true)

![Story](https://github.com/jaechanjo/Yolov7_TensorRT_Triton/blob/main/docs/images/Story.PNG?raw=true)

![Example](https://github.com/jaechanjo/Yolov7_TensorRT_Triton/blob/main/docs/images/Example.PNG?raw=true)

## How to Inference

더 자세한 실행 방법은 다음의 [PPT](https://drive.google.com/file/d/10lXwgwR-QrleC1RGkFpeepzrdf8dZWNA/view?usp=drive_link) 문서를 참고하세요.
