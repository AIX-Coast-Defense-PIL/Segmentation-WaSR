## Segmentation
---
#### WaSR: A Water Segmentation and Refinement Maritime Obstacle Detection Network
[[`github`](https://github.com/lojzezust/WaSR)] 
[[`paper`](https://prints.vicos.si/publications/392/wasr-a-water-segmentation-and-refinement-maritime-obstacle-detection-network)] [[`original implementation`](https://github.com/bborja/wasr_network)] 
[[`BibTeX`](#cite)] 

```bash
## train (background)
. shell/train_wasr_bg.sh &

## inference
. shell/infer_wasr.sh 
# 참고: mode==pred -> results: images (seaships와 같이 정성적 평가만 가능할 때)
# 참고: mode==eval -> results: images, accuracy, iou (aihub와 같이 정량적 평가도 가능할 때)
```

<br/>

## Object Detection
---
#### YOLOv5
[[`github`](https://github.com/ultralytics/yolov5)] 