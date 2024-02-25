import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_curve(files=[], legends=[], formats=[], type='ap50', epoch=50):
    if type == 'ap50':
        for file, legend, format in zip(files, legends, formats):
            if format == 'mmdet_voc':
                maps = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if '(val)' in line:
                        map = float(line.split(':')[-2].split(',')[0])
                        maps.append(map)

            elif format == 'yolo':
                maps = pd.read_csv(file)['     metrics/mAP_0.5']
            plt.plot(maps[:epoch], label=legend)
            plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel('mAP_0.5')
        plt.show()

    if type == 'map':
        for file, legend, format in zip(files, legends, formats):
            if format == 'mmdet_voc':
                maps = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if '(val)' in line:
                        map = float(line.split(':')[-2].split(',')[0])
                        maps.append(map)

            elif format == 'yolo':
                maps = pd.read_csv(file)['     metrics/mAP_0.5']

            elif format == 'mmdet_bbox':
                maps = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if 'Epoch(val)' in line:
                        line = line.split(',')
                        map = float(line[1].split(':')[-1])
                        maps.append(map)
            elif format == 'mmdet_seg':
                maps = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if 'Epoch(val)' in line:
                        line = line.split(',')
                        map = float(line[-7].split(':')[-1])
                        maps.append(map)
            plt.plot(maps[:epoch], label=legend)
            plt.legend(loc='best')

        if format == 'mmdet_seg':
            plt.xlabel('epoch')
            plt.ylabel('Segm_mAP')
            plt.savefig('Segm_mAP.png')
#             plt.show()
        else:
            plt.xlabel('epoch')
            plt.ylabel('Bbox_mAP')
            plt.savefig('Bbox_mAP.png')

#             plt.show()



    if type == 'loss':
        for file, legend, format in zip(files, legends, formats):
            if format == 'mmdet_voc':
                loss_temp = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if 'loss:' in line:
                        loss = float(line.split('loss:')[1].split(',')[0])
                        loss_temp.append(loss)
                index = np.linspace(0, len(loss_temp)-1, num=epoch, dtype=int)
                loss_all = [loss_temp[indx] for indx in index]

            elif format == 'yolo':
                logs = pd.read_csv(file)
                loss_all = logs['      train/box_loss'] + logs['      train/box_loss'] + logs['      train/cls_loss']
            elif format == 'mmdet_bbox':
                loss_temp = []
                with open(file, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'Evaluating bbox' in line:
                        line = lines[i-2]
                        loss = float(line.split('loss:')[1])
                        loss_temp.append(loss)
                index = np.linspace(0, len(loss_temp)-1, num=epoch, dtype=int)
                loss_all = [loss_temp[indx] for indx in index]
            x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            plt.plot(x, loss_all[:epoch])
#             plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('loss.png')
#         plt.show()

if __name__ == '__main__':

    # 文件地址、图例、格式
#     files = ['work_dirs/mask2former_r50_lsj_8x2_1x_qiangji/20230601_184459.log',
#              'work_dirs/solov2_r50_fpn_1x_qiangji/20230602_073947.log',
#            'work_dirs/cascade_mask_rcnn_r50_fpn_1x_qiangji/20230522_011331.log',
#              'work_dirs/mask_rcnn_r50_fpn_1x_qiangji/20230529_002221.log',
#              'work_dirs/mask_rcnn_r50_psafpn_1x_qiangji-s=8/20230530_020839.log'
#              ]
#     legends = ['Mask2Former',
#                'SOLOV2',
#                'Cascade-MaskRCNN',
#                'MaskRCNN',
#                'PSA-MaskRCNN'
#                ]
    files = [
            'work_dirs/logs/mask_rcnn.log',
            'work_dirs/logs/mask_rcnn_psar50.log',
            'work_dirs/logs/cascade_mask_rcnn.log',
            'work_dirs/logs/cascade_mask_rcnn_psa.log',
            'work_dirs/logs/groie.log',
            'work_dirs/logs/groie_psa.log',
            'work_dirs/logs/point_rend.log',
            'work_dirs/logs/point_rend_psa.log',
                 ]
    legends = [
            'MaskRCNN',
            'PSA-MaskRCNN',
            'CascadeRCNN',
            'PSA-CascadeRCNN',
            'Groie',
            'PSA-Groie',
            'PointRend',
            'PSA-PointRend',
            
    ]
    fomats2 = ['mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   'mmdet_seg',
                   ]
    fomats1 = ['mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   'mmdet_bbox',
                   ]
#     plot_curve(files, legends, fomats1, 'loss', epoch=12)
#     plot_curve(files, legends, fomats2, 'map', epoch=200)
    plot_curve(files, legends, fomats2, 'map', epoch=12)
