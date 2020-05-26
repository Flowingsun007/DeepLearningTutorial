import torch
from torch.autograd import Variable
import numpy as np
import cv2

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from matplotlib import pyplot as plt
from data import VOC_CLASSES as labels


def test(in_path, weight_path, out_path='data/demo.jpg'):
    # read image
    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()

    # build model
    net = build_ssd('test', 300, 21)  # initialize SSD
    net.load_weights(weight_path)

    # detect image
    with torch.no_grad():
        y = net(xx)

    plt.figure(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)  # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            score = detections[0, i, j, 0]
            label_name = labels[i - 1]
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale)
            # pt是tensor列表[x1, x2, y1, y2],需转化成普通列表pt_list
            pt_list = np.round(pt.cpu().numpy()).astype(np.int).tolist()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            # put boxes and label test into image(for cv2.imwrite)
            cv2.rectangle(img=image, pt1=(pt_list[0], pt_list[1]), pt2=(pt_list[2], pt_list[3]),
                          color=(255, 0, 0), thickness=2)
            cv2.putText(img=image, text=display_txt, org=(pt_list[0], pt_list[1] - 10),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=1)
            j += 1

    # show plot
    plt.show()
    # save detected image
    cv2.imwrite(out_path, image)
    print('image saved in: ' + out_path)


if __name__ == '__main__':
    img_path = './data/demo.jpg'
    model_path = './weights/ssd300_mAP_77.43_v2.pth'
    test(img_path, model_path)
