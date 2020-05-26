**[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)** 的修改版实现，模型结构，损失函数和功能都和原版本保持一致，除此之外，在原版本的基础上做了如下改动：


- **1.去掉了原来训练时的visdom可视化，改成了tensorboard**
- **2.修复了原版本的部分bug,能正常在pytorch1.0,1.3,1.4运行train,test,eval**
- **3.改动了原test.py，现在的test.py支持单张图片检测、预览和保存结果**
- **4.删除了部分无用代码；添加了部分注释，更容易阅读**



**[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)**'s another implementation, which has same model structure with origin,but I made some changes:
**1.remove visdom,change to tensorboard**
**2.fix some bug,now it can normally train/test/eval with pytorch1.0,1.3,1.4**
**3.modify test.py, now it can detect single image and save detected result**
**3.delete some useless codes;add some annotation**

# 1.test
[download pre-trained model](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth) and put it in weights/ssd300_mAP_77.43_v2.pth
```python
python test.py
```
![Figure_1.png](https://cdn.nlark.com/yuque/0/2020/png/216914/1590473566464-bf3a951f-bf2b-48ef-8b24-cfceb29bef19.png#align=left&display=inline&height=818&margin=%5Bobject%20Object%5D&name=Figure_1.png&originHeight=818&originWidth=1090&size=772757&status=done&style=none&width=1090)
# 2.train
before training, you should prepare dataset(like voc) first and set VOC_ROOT in voc0712.py,then download vgg pre-train weight:
https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
,and put it in weights/vgg16_reducedfc.pth

```python
python train.py
# show training logs
tensorboard --logdir data/logs
```
Output:
```python
timer: 5.9624 sec.
lr: 0.001 || iter: 0 || box_loss: 2.9634 || conf loss: 22.2013 || total loss: 25.1646 || timer: 0.2626 sec.
lr: 0.001 || iter: 10 || box_loss: 2.9099 || conf loss: 13.7757 || total loss: 16.6855 || timer: 0.2643 sec.
lr: 0.001 || iter: 20 || box_loss: 3.0165 || conf loss: 13.1046 || total loss: 16.1211 || timer: 0.2588 sec.
lr: 0.001 || iter: 30 || box_loss: 2.8122 || conf loss: 10.7875 || total loss: 13.5996 || timer: 0.2597 sec.
lr: 0.001 || iter: 40 || box_loss: 3.0991 || conf loss: 8.7338 || total loss: 11.8330 || timer: 0.2591 sec.
lr: 0.001 || iter: 50 || box_loss: 2.7000 || conf loss: 8.0487 || total loss: 10.7487 || timer: 0.2634 sec.
lr: 0.001 || iter: 60 || box_loss: 2.7865 || conf loss: 7.2490 || total loss: 10.0355 || timer: 0.2633 sec.
lr: 0.001 || iter: 70 || box_loss: 2.6665 || conf loss: 6.5484 || total loss: 9.2149 || timer: 0.2613 sec.
lr: 0.001 || iter: 80 || box_loss: 2.8356 || conf loss: 6.2455 || total loss: 9.0810 || timer: 0.2614 sec.
lr: 0.001 || iter: 90 || box_loss: 2.9678 || conf loss: 6.2002 || total loss: 9.1680 || timer: 0.2612 sec.
lr: 0.001 || iter: 100 || box_loss: 2.6681 || conf loss: 6.0221 || total loss: 8.6902 || timer: 0.2623 sec.
```
# 3.eval
[download pre-trained model](https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth) and put it in weights/ssd300_mAP_77.43_v2.pth
```python
python eval.py
```
Output:
```python
im_detect: 4950/4952 0.018s
im_detect: 4951/4952 0.054s
im_detect: 4952/4952 0.046s
Evaluating detections
Writing aeroplane VOC results file
Writing bicycle VOC results file
Writing bird VOC results file
Writing boat VOC results file
Writing bottle VOC results file
Writing bus VOC results file
Writing car VOC results file
Writing cat VOC results file
Writing chair VOC results file
Writing cow VOC results file
Writing diningtable VOC results file
Writing dog VOC results file
Writing horse VOC results file
Writing motorbike VOC results file
Writing person VOC results file
Writing pottedplant VOC results file
Writing sheep VOC results file
Writing sofa VOC results file
Writing train VOC results file
Writing tvmonitor VOC results file
VOC07 metric? Yes
AP for aeroplane = 0.8207
AP for bicycle = 0.8568
AP for bird = 0.7546
AP for boat = 0.6952
AP for bottle = 0.5019
AP for bus = 0.8479
AP for car = 0.8584
AP for cat = 0.8734
AP for chair = 0.6136
AP for cow = 0.8243
AP for diningtable = 0.7906
AP for dog = 0.8566
AP for horse = 0.8714
AP for motorbike = 0.8403
AP for person = 0.7895
AP for pottedplant = 0.5069
AP for sheep = 0.7767
AP for sofa = 0.7894
AP for train = 0.8623
AP for tvmonitor = 0.7670
Mean AP = 0.7749
~~~~~~~~
Results:
0.821
0.857
0.755
0.695
0.502
0.848
0.858
0.873
0.614
0.824
0.791
0.857
0.871
0.840
0.790
0.507
0.777
0.789
0.862
0.767
0.775
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
--------------------------------------------------------------
```

