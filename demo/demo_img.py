import torch.cuda

from mmdet.apis import init_detector, inference_detector


def main():
    # 根据配置文件和checkpoint文件构建模型
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
    # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/
    # faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:0'
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device=device)
    # 测试单张图片并生成推理结果
    img = 'demo/files/000000581897.jpg'
    result = inference_detector(model, img)

    # 在一个新的窗口中将结果可视化
    # model.show_result(img,result)
    # 或者将可视化结果保存为图片
    model.show_result(img, result, out_file="demo/output/000000581897.jpg")


if __name__ == '__main__':
    # python demo/demo_img.py
    print(torch.cuda.is_available())
    print(torch.cuda.get_arch_list())
    main()
    # datum convert --input-format voc --input-path data/VOC2012 --output-format coco --output-dir data/coco_voc2012_cat --filter '/item[annotation/label="cat"]'
    # python train.py
    # --batch_size 48
    # --save_folder ../work_output/
    # --image_sets datasets/HiXray/train/train_name.txt
    # --transfer ../checkpoints/ssd300_mAP_77.43_v2.pth
