# Projeto Final - Modelos Preditivos Conexionistas

### Gabriel de Albuquerque Araújo

|**Tipo de Projeto**|**Modelo Selecionado**|**Linguagem**|
|--|--|--|
|Deteção de Objetos|YOLOv5|PyTorch|

## Performance

O modelo treinado possui performance de 99,09%.

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```text
     wandb: Currently logged in as: gaa2. Use `wandb login --relogin` to force relogin
train: weights=yolov5x.pt, cfg=, data=/content/yolov5/Reconhecimento-de-Capacete-3/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=4000, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-8-g6728dad Python-3.7.13 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
wandb: Tracking run with wandb version 0.13.1
wandb: Run data is saved locally in /content/yolov5/wandb/run-20220817_215741-2sbg63x1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run sweet-jazz-5
wandb: ⭐️ View project at https://wandb.ai/gaa2/YOLOv5
wandb: 🚀 View run at https://wandb.ai/gaa2/YOLOv5/runs/2sbg63x1
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
100% 755k/755k [00:00<00:00, 17.0MB/s]
YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected.
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt to yolov5x.pt...
100% 166M/166M [00:02<00:00, 65.7MB/s]

Overriding model.yaml nc=80 with nc=1

                 from  n    params  module                                  arguments                     
  0                -1  1      8800  models.common.Conv                      [3, 80, 6, 2, 2]              
  1                -1  1    115520  models.common.Conv                      [80, 160, 3, 2]               
  2                -1  4    309120  models.common.C3                        [160, 160, 4]                 
  3                -1  1    461440  models.common.Conv                      [160, 320, 3, 2]              
  4                -1  8   2259200  models.common.C3                        [320, 320, 8]                 
  5                -1  1   1844480  models.common.Conv                      [320, 640, 3, 2]              
  6                -1 12  13125120  models.common.C3                        [640, 640, 12]                
  7                -1  1   7375360  models.common.Conv                      [640, 1280, 3, 2]             
  8                -1  4  19676160  models.common.C3                        [1280, 1280, 4]               
  9                -1  1   4099840  models.common.SPPF                      [1280, 1280, 5]               
 10                -1  1    820480  models.common.Conv                      [1280, 640, 1, 1]             
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  4   5332480  models.common.C3                        [1280, 640, 4, False]         
 14                -1  1    205440  models.common.Conv                      [640, 320, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  4   1335040  models.common.C3                        [640, 320, 4, False]          
 18                -1  1    922240  models.common.Conv                      [320, 320, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  4   4922880  models.common.C3                        [640, 640, 4, False]          
 21                -1  1   3687680  models.common.Conv                      [640, 640, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  4  19676160  models.common.C3                        [1280, 1280, 4, False]        
 24      [17, 20, 23]  1     40374  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [320, 640, 1280]]
Model summary: 567 layers, 86217814 parameters, 86217814 gradients, 204.6 GFLOPs

Transferred 739/745 items from yolov5x.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 123 weight(decay=0.0), 126 weight(decay=0.0005), 126 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/yolov5/Reconhecimento-de-Capacete-3/train/labels' images and labels...204 found, 0 missing, 54 empty, 0 corrupt: 100% 204/204 [00:00<00:00, 2222.07it/s]
train: New cache created: /content/yolov5/Reconhecimento-de-Capacete-3/train/labels.cache
train: Caching images (0.3GB ram): 100% 204/204 [00:01<00:00, 148.46it/s]
val: Scanning '/content/yolov5/Reconhecimento-de-Capacete-3/valid/labels' images and labels...19 found, 0 missing, 5 empty, 0 corrupt: 100% 19/19 [00:00<00:00, 573.38it/s]
val: New cache created: /content/yolov5/Reconhecimento-de-Capacete-3/valid/labels.cache
val: Caching images (0.0GB ram): 100% 19/19 [00:00<00:00, 71.45it/s]
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 5.04 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to runs/train/exp
Starting training for 4000 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    0/3999     13.6G    0.1131   0.02566         0        24       640: 100% 13/13 [00:19<00:00,  1.52s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:02<00:00,  2.11s/it]
                 all         19         18    0.00158        0.5    0.00454   0.000946

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    1/3999     14.2G   0.09572   0.02677         0        15       640: 100% 13/13 [00:13<00:00,  1.02s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.49it/s]
                 all         19         18    0.00211      0.667     0.0124    0.00411

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    2/3999     14.2G    0.0805   0.02588         0        21       640: 100% 13/13 [00:13<00:00,  1.02s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.72it/s]
                 all         19         18    0.00211      0.667     0.0202    0.00696

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    3/3999     14.2G   0.07553   0.02249         0        12       640: 100% 13/13 [00:13<00:00,  1.03s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.77it/s]
                 all         19         18    0.00298      0.944     0.0995     0.0312

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    4/3999     14.2G   0.07069   0.02535         0        23       640: 100% 13/13 [00:13<00:00,  1.04s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.78it/s]
                 all         19         18    0.00522          1      0.116     0.0529

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    5/3999     14.2G   0.06615   0.02321         0        17       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18     0.0245      0.778      0.249     0.0738

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    6/3999     14.2G   0.06151   0.02458         0        12       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.81it/s]
                 all         19         18      0.182        0.5      0.376      0.108

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    7/3999     14.2G   0.05853   0.02256         0        19       640: 100% 13/13 [00:14<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.82it/s]
                 all         19         18      0.513      0.389       0.47      0.188

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    8/3999     14.2G   0.06132   0.02154         0        18       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.83it/s]
                 all         19         18      0.355      0.389      0.403      0.153

     Epoch   gpu_mem       box       obj       cls    labels  img_size
    9/3999     14.2G     0.053   0.02091         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.80it/s]
                 all         19         18      0.213      0.556      0.285      0.164

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   10/3999     14.2G   0.04903   0.02248         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.83it/s]
                 all         19         18      0.406      0.418       0.48      0.168

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   11/3999     14.2G   0.05142   0.02046         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.556      0.444      0.502      0.176

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   12/3999     14.2G   0.04879   0.01933         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.611      0.389      0.643      0.175

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   13/3999     14.2G   0.04739   0.01939         0        30       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.468      0.667      0.588      0.341

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   14/3999     14.2G   0.04703   0.01795         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18       0.52      0.667      0.617      0.258

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   15/3999     14.2G    0.0472   0.01712         0        18       640: 100% 13/13 [00:14<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.421      0.722      0.475      0.127

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   16/3999     14.2G   0.04491   0.01715         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.752      0.444      0.615      0.267

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   17/3999     14.2G    0.0457   0.01631         0        24       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.673      0.802      0.739      0.237

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   18/3999     14.2G   0.04216   0.01476         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.486      0.473      0.363      0.103

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   19/3999     14.2G   0.04491   0.01444         0        22       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.94it/s]
                 all         19         18      0.709      0.556      0.692      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   20/3999     14.2G   0.03961   0.01506         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.453      0.667      0.477      0.153

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   21/3999     14.2G   0.04019   0.01331         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18       0.75      0.722      0.784      0.308

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   22/3999     14.2G   0.03806   0.01505         0        18       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.738      0.611      0.712      0.361

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   23/3999     14.2G   0.03997   0.01332         0        22       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.656      0.667      0.656      0.263

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   24/3999     14.2G   0.03865   0.01341         0        24       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18       0.72      0.667      0.754      0.386

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   25/3999     14.2G   0.03748   0.01243         0        19       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.663      0.667      0.669      0.202

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   26/3999     14.2G   0.03881    0.0129         0        26       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.853      0.444      0.566      0.191

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   27/3999     14.2G   0.03793   0.01258         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.794      0.722      0.746      0.412

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   28/3999     14.2G   0.03619   0.01251         0        24       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.779      0.833      0.758      0.335

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   29/3999     14.2G   0.03544   0.01194         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.903      0.722      0.795      0.472

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   30/3999     14.2G   0.03661   0.01129         0        27       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.627      0.722      0.661      0.324

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   31/3999     14.2G   0.03351   0.01146         0        23       640: 100% 13/13 [00:13<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.806      0.695      0.775      0.464

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   32/3999     14.2G   0.03517   0.01062         0        15       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.727      0.778      0.793      0.328

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   33/3999     14.2G   0.03474    0.0123         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.78it/s]
                 all         19         18      0.841      0.667      0.798      0.378

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   34/3999     14.2G   0.03306   0.01138         0        30       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.866      0.718      0.753      0.437

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   35/3999     14.2G   0.03614   0.01056         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18       0.82      0.611      0.688      0.279

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   36/3999     14.2G   0.03343   0.01183         0        18       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18       0.81      0.711      0.818      0.389

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   37/3999     14.2G   0.03344   0.01205         0        33       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.888      0.556      0.659      0.276

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   38/3999     14.2G    0.0315  0.009562         0        13       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.731      0.667      0.732      0.458

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   39/3999     14.2G   0.03447   0.01194         0        39       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.639      0.787      0.744      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   40/3999     14.2G   0.03338   0.01101         0        31       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.751      0.671        0.8      0.431

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   41/3999     14.2G   0.03384  0.009346         0        25       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.745      0.833      0.868      0.353

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   42/3999     14.2G   0.03025   0.01017         0        22       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.811      0.717      0.788      0.276

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   43/3999     14.2G   0.03316  0.009604         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18       0.84      0.667      0.669      0.325

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   44/3999     14.2G     0.032   0.01011         0        18       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.819      0.778      0.783      0.361

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   45/3999     14.2G   0.03147   0.01126         0        30       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.78it/s]
                 all         19         18      0.659      0.754      0.737      0.353

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   46/3999     14.2G   0.03024   0.01015         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.785        0.5      0.597      0.246

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   47/3999     14.2G   0.03123  0.009573         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.675      0.722      0.693      0.362

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   48/3999     14.2G   0.03122  0.009875         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.719      0.667      0.638      0.245

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   49/3999     14.2G   0.02952  0.009431         0        34       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.872      0.667      0.765      0.431

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   50/3999     14.2G   0.02964  0.008555         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.732      0.722       0.75      0.378

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   51/3999     14.2G   0.03055  0.009261         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.922       0.66      0.788      0.358

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   52/3999     14.2G    0.0306  0.009588         0        26       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.916      0.604      0.722      0.273

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   53/3999     14.2G   0.02849   0.01019         0        32       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.83it/s]
                 all         19         18      0.961      0.611      0.825       0.39

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   54/3999     14.2G   0.02898  0.009487         0        10       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.925       0.69      0.897      0.521

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   55/3999     14.2G   0.02903  0.008175         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.834      0.722      0.841      0.344

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   56/3999     14.2G   0.02815  0.009018         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.746      0.833      0.855      0.432

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   57/3999     14.2G   0.03034  0.008804         0        14       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.799      0.661      0.678      0.269

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   58/3999     14.2G   0.02982  0.009763         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.978      0.667      0.787      0.342

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   59/3999     14.2G   0.02981   0.00903         0        26       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.761      0.709      0.714      0.432

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   60/3999     14.2G   0.03089   0.00899         0        18       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.916      0.605      0.766      0.309

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   61/3999     14.2G   0.02604  0.008808         0        18       640: 100% 13/13 [00:14<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.899      0.556      0.779      0.358

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   62/3999     14.2G   0.02886  0.008224         0        22       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.991      0.667      0.792      0.306

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   63/3999     14.2G   0.02834  0.008504         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.747      0.656      0.706      0.296

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   64/3999     14.2G   0.02824  0.008214         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18       0.67      0.722      0.726      0.383

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   65/3999     14.2G   0.02841  0.008728         0        24       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.673      0.667      0.747      0.221

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   66/3999     14.2G   0.02807  0.008169         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.924      0.681      0.842       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   67/3999     14.2G   0.02749  0.008767         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.812      0.722       0.75      0.288

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   68/3999     14.2G   0.02652   0.00761         0        19       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.723      0.889      0.835      0.327

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   69/3999     14.2G   0.02717  0.008167         0        30       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.869      0.778      0.721      0.429

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   70/3999     14.2G   0.02827   0.00843         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.705      0.797      0.834      0.346

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   71/3999     14.2G   0.02613  0.007998         0        15       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.834      0.778      0.869       0.42

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   72/3999     14.2G   0.02556  0.008248         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.797      0.874      0.869      0.417

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   73/3999     14.2G   0.02528   0.00802         0        22       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.869      0.722      0.854      0.392

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   74/3999     14.2G   0.02651  0.008511         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.806      0.778      0.842      0.274

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   75/3999     14.2G   0.02745  0.008547         0        18       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.687      0.833      0.815      0.404

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   76/3999     14.2G   0.02634  0.008306         0        22       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.617      0.667       0.61        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   77/3999     14.2G   0.02563   0.00697         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.764      0.721      0.699      0.346

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   78/3999     14.2G   0.02502  0.007662         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.779      0.722      0.715      0.426

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   79/3999     14.2G     0.027   0.00818         0        26       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.667      0.833       0.76      0.481

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   80/3999     14.2G   0.02769  0.008375         0        19       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.822      0.772      0.796      0.353

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   81/3999     14.2G     0.026  0.009136         0        25       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.958      0.667      0.835      0.349

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   82/3999     14.2G   0.02584  0.007662         0        31       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.978      0.722      0.852      0.442

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   83/3999     14.2G   0.02568  0.007751         0        14       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.882      0.722      0.824      0.463

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   84/3999     14.2G   0.02673  0.008889         0        29       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.928       0.72      0.805      0.276

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   85/3999     14.2G   0.02418  0.007714         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.983      0.667      0.801      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   86/3999     14.2G   0.02287  0.008161         0        16       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.667      0.669      0.731      0.279

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   87/3999     14.2G   0.02491  0.007923         0        25       640: 100% 13/13 [00:14<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.616        0.5      0.621      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   88/3999     14.2G   0.02543  0.008189         0        23       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.752      0.674        0.7      0.291

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   89/3999     14.2G   0.02529  0.008036         0        22       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.821      0.667      0.723      0.422

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   90/3999     14.2G   0.02445  0.007227         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.914      0.722      0.777      0.434

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   91/3999     14.2G    0.0257  0.007763         0        13       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.683      0.667      0.739      0.329

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   92/3999     14.2G   0.02473  0.007596         0        27       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.755      0.889      0.832      0.406

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   93/3999     14.2G   0.02349  0.007433         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.726      0.889       0.84      0.424

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   94/3999     14.2G   0.02286  0.007125         0        21       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.786      0.833      0.858      0.326

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   95/3999     14.2G   0.02599  0.007634         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.919      0.778      0.816      0.403

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   96/3999     14.2G   0.02312  0.007747         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.93it/s]
                 all         19         18      0.759      0.667      0.709      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   97/3999     14.2G   0.02271  0.007059         0        18       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.673      0.722      0.676      0.384

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   98/3999     14.2G   0.02178  0.007728         0        29       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.666      0.611      0.615        0.3

     Epoch   gpu_mem       box       obj       cls    labels  img_size
   99/3999     14.2G   0.02484  0.007287         0        25       640: 100% 13/13 [00:13<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.839      0.579      0.636      0.425

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  100/3999     14.2G   0.02486  0.008142         0        26       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.693      0.667      0.625      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  101/3999     14.2G   0.02247  0.007173         0        29       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.613      0.667      0.608      0.247

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  102/3999     14.2G   0.02228  0.007177         0        15       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.80it/s]
                 all         19         18      0.642      0.722      0.602      0.286

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  103/3999     14.2G   0.02258   0.00699         0        27       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.605      0.596      0.587       0.28

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  104/3999     14.2G   0.02206  0.007568         0        38       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18       0.65      0.721      0.694      0.389

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  105/3999     14.2G   0.02156  0.007109         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.678      0.556      0.689      0.338

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  106/3999     14.2G    0.0224  0.007404         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.737      0.611       0.72      0.364

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  107/3999     14.2G    0.0245  0.007323         0        11       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.809      0.667      0.739      0.365

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  108/3999     14.2G   0.02471  0.006623         0        21       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.652      0.611      0.612      0.263

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  109/3999     14.2G   0.02042  0.007419         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18       0.67      0.722      0.649       0.37

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  110/3999     14.2G   0.02085  0.006368         0        13       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18       0.78      0.593      0.643      0.301

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  111/3999     14.2G   0.02255  0.007283         0        27       640: 100% 13/13 [00:14<00:00,  1.08s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.712      0.389      0.539      0.251

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  112/3999     14.2G   0.02067  0.006721         0        26       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.659      0.444      0.598      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  113/3999     14.2G    0.0208  0.007254         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.736      0.775      0.822      0.305

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  114/3999     14.2G   0.02041  0.007087         0        15       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.724      0.877      0.846      0.397

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  115/3999     14.2G    0.0237  0.006989         0        15       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18       0.73      0.722      0.791      0.258

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  116/3999     14.2G   0.02279  0.006912         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.74it/s]
                 all         19         18      0.714      0.611      0.675      0.259

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  117/3999     14.2G   0.02241  0.007062         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.721      0.574      0.648      0.208

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  118/3999     14.2G   0.02091  0.007167         0        24       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.735        0.5       0.52      0.213

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  119/3999     14.2G   0.02083  0.006885         0        18       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.724      0.437      0.457      0.205

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  120/3999     14.2G   0.02194  0.007186         0        18       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.674      0.389      0.436      0.179

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  121/3999     14.2G   0.02321  0.007084         0        14       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.748      0.389      0.485      0.232

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  122/3999     14.2G   0.02172  0.007434         0        24       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.72it/s]
                 all         19         18      0.584      0.556      0.531      0.231

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  123/3999     14.2G   0.02165  0.006252         0        24       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.545      0.534       0.54      0.302

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  124/3999     14.2G   0.02309  0.006813         0        18       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.466      0.387       0.45      0.212

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  125/3999     14.2G   0.02178   0.00663         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18       0.57       0.59      0.545      0.295

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  126/3999     14.2G    0.0221  0.006263         0        15       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.708      0.722      0.674      0.397

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  127/3999     14.2G   0.02241  0.006331         0        19       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18      0.693      0.667      0.572      0.263

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  128/3999     14.2G   0.02283  0.006384         0        30       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.773      0.611      0.705      0.347

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  129/3999     14.2G    0.0197  0.006844         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.778      0.611      0.631      0.243

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  130/3999     14.2G   0.02059  0.006605         0        25       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.86it/s]
                 all         19         18      0.772      0.444      0.525      0.224

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  131/3999     14.2G   0.02129  0.006737         0        29       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.93it/s]
                 all         19         18      0.807      0.556      0.664      0.284

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  132/3999     14.2G   0.02258  0.007372         0        30       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.91it/s]
                 all         19         18      0.914      0.591      0.732      0.427

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  133/3999     14.2G   0.01943  0.006421         0        20       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.922      0.654      0.751      0.428

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  134/3999     14.2G   0.02076  0.005952         0        24       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.905      0.556      0.667      0.367

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  135/3999     14.2G   0.02192   0.00623         0        14       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.926      0.556      0.648       0.31

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  136/3999     14.2G   0.02179  0.006128         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.814      0.556      0.686      0.381

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  137/3999     14.2G   0.02054  0.007008         0        27       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.84it/s]
                 all         19         18      0.719      0.722      0.722      0.463

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  138/3999     14.2G   0.01986  0.006715         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.851      0.667       0.78      0.447

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  139/3999     14.2G   0.02206  0.006512         0        31       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18       0.77      0.778      0.794      0.416

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  140/3999     14.2G   0.02183   0.00642         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.808      0.833      0.859      0.503

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  141/3999     14.2G   0.02147   0.00632         0        17       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.847      0.722      0.822      0.379

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  142/3999     14.2G   0.01884  0.006309         0        16       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.901      0.556      0.726       0.37

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  143/3999     14.2G   0.01952  0.006286         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.882      0.556      0.675      0.407

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  144/3999     14.2G   0.01996     0.007         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.83it/s]
                 all         19         18      0.892      0.556      0.658       0.36

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  145/3999     14.2G   0.01976   0.00625         0        29       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18      0.895      0.611      0.663       0.34

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  146/3999     14.2G   0.01814  0.006381         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.834      0.611      0.647       0.32

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  147/3999     14.2G   0.02089  0.006064         0        28       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.89it/s]
                 all         19         18       0.91      0.722      0.814      0.438

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  148/3999     14.2G   0.02182  0.006406         0        24       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.88it/s]
                 all         19         18      0.915      0.722      0.825      0.439

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  149/3999     14.2G    0.0207  0.006961         0        26       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.812      0.722      0.818      0.518

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  150/3999     14.2G   0.02069  0.007037         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.87it/s]
                 all         19         18      0.906      0.722      0.832      0.435

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  151/3999     14.2G   0.02024  0.006582         0        20       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.92it/s]
                 all         19         18      0.858      0.722      0.777       0.34

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  152/3999     14.2G   0.02185  0.006424         0        17       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18          1      0.435      0.633      0.342

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  153/3999     14.2G   0.01958  0.006888         0        23       640: 100% 13/13 [00:13<00:00,  1.07s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.90it/s]
                 all         19         18          1      0.496       0.61      0.322

     Epoch   gpu_mem       box       obj       cls    labels  img_size
  154/3999     14.2G   0.01917  0.005312         0        13       640: 100% 13/13 [00:13<00:00,  1.06s/it]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.85it/s]
                 all         19         18      0.799      0.664       0.73      0.404
Stopping training early as no improvement observed in last 100 epochs. Best results observed at epoch 54, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.

155 epochs completed in 0.752 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 173.2MB
Optimizer stripped from runs/train/exp/weights/best.pt, 173.2MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model summary: 444 layers, 86173414 parameters, 0 gradients, 203.8 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  1.68it/s]
                 all         19         18      0.925      0.689      0.897      0.522
Results saved to runs/train/exp
wandb: Waiting for W&B process to finish... (success).
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:      metrics/mAP_0.5 ▁▂▅▅▅▆▆▇▇▆▇▆▆▇█▇▆▇█▇▇▇▆▇▇▆▆▇▅▇▅▅▅▆▆▇▆▇▇█
wandb: metrics/mAP_0.5:0.95 ▁▁▃▃▃▅▅▇▇▅▅▅▆▆▆▇▅▅▇▆▇▇▆▅▆▇▅▆▄▄▄▅▅▅▅▇▆▇▆█
wandb:    metrics/precision ▁▁▅▅▄▆▆▇▇▇▆▇▆█▇▇▇▇▇▆▆█▆▆█▇▆▇▆▇▆▅▆▇█▇██▇█
wandb:       metrics/recall ▅█▁▂▅▃▅▅▅▄▆▅▅▄▅▅▄▅▆▇▇▅▂▅▆▃▄▅▁▅▂▃▅▃▃▆▃▅▅▅
wandb:       train/box_loss █▆▅▄▄▃▃▃▂▃▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/cls_loss ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:       train/obj_loss █▇▇▆▅▄▃▃▃▂▃▂▂▂▂▂▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         val/box_loss █▅▄▃▅▃▄▂▂▃▃▂▃▂▂▂▂▂▁▂▁▂▂▂▂▁▃▂▂▃▃▂▂▃▃▂▂▂▃▂
wandb:         val/cls_loss ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         val/obj_loss █▆▅▃▂▂▁▁▁▁▂▃▂▂▂▂▂▁▁▁▂▁▃▂▁▃▄▃▅▂▆▆▄▃▄▂▃▂▂▂
wandb:                x/lr0 █▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                x/lr1 ▁▃██████████████████████████████████████
wandb:                x/lr2 ▁▃██████████████████████████████████████
wandb: 
wandb: Run summary:
wandb:           best/epoch 54
wandb:         best/mAP_0.5 0.89721
wandb:    best/mAP_0.5:0.95 0.52149
wandb:       best/precision 0.92535
wandb:          best/recall 0.68964
wandb:      metrics/mAP_0.5 0.89725
wandb: metrics/mAP_0.5:0.95 0.52151
wandb:    metrics/precision 0.92531
wandb:       metrics/recall 0.68929
wandb:       train/box_loss 0.01917
wandb:       train/cls_loss 0.0
wandb:       train/obj_loss 0.00531
wandb:         val/box_loss 0.03414
wandb:         val/cls_loss 0.0
wandb:         val/obj_loss 0.00803
wandb:                x/lr0 0.00962
wandb:                x/lr1 0.00962
wandb:                x/lr2 0.00962
wandb: 
wandb: Synced sweet-jazz-5: https://wandb.ai/gaa2/YOLOv5/runs/2sbg63x1
wandb: Synced 5 W&B file(s), 13 media file(s), 1 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20220817_215741-2sbg63x1/logs
  ```
</details>

### Evidências do treinamento

Nessa seção você deve colocar qualquer evidência do treinamento, como por exemplo gráficos de perda, performance, matriz de confusão etc.

Exemplo de adição de imagem:
![Descrição](blob:https://wandb.ai/482c63ee-670c-423e-bfc9-988d6034c684)

## Roboflow

Nessa seção deve colocar o link para acessar o dataset no Roboflow

[Link RoboFlow](https://app.roboflow.com/cesar-school/reconhecimento-de-capacete/overview)

## HuggingFace

[Link HuggingFace](https://huggingface.co/Gabriel-albuq/Projeto1)