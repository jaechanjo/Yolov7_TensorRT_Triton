# Yolov7_TensorRT_Triton
  - Lightweight and optimization experiment using Nvidia TensorRT version 8.0.1 for yolov7 pytorch model
  - Deploy yolov7 tensorrt model on 21.08 Triton Server

## Performance

![Performance](docs/images/Performance)

## Project Introduction

  1. YOLOv7, 가장 최신 버전의 Object Detection 모델 적용
  
  
    - TAO Toolkit (current version, 4.0기준) : YOLOv4 업데이트
  2. CCTV 이상 행동 중 실신 탐지 가능
  
  
    - 기존 MSCOCO 사전학습 모델은 Standing Person만 탐지 가능
  3. 실생활 적용을 위한 실시간 구동 추론 속도 확인
  
  
    - Triton Server 기준 94.4 infer/sec(16 concurrency)

![Pipeline](docs/images/Pipeline)

![Story](docs/images/Story)

![Example](docs/images/Example)

## How to Inference

더 자세한 실행 방법은 아래의 PPT 문서를 참고하세요.

[CCTV Anomaly Detection Real-time Event Detection System.ppt]()
