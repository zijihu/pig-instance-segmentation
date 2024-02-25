import mmcv
import numpy as np

path = 'work_dirs/mask_rcnn_r50_fpn_3x_mycoco/results.pkl'
results = mmcv.load(path)
for i, result in enumerate(results):
    shuye_scores = result[0][0]
    bingban_scores = result[0][1]
    shuyes = result[1][0]
    bingbans = result[1][1]

    shuye_area = 0
    bingban_area = 0
    for shuye_score, shuye in zip(shuye_scores, shuyes):
        if shuye_score[-1] < 0.5:
            continue
        temp = shuye['counts']
        temp = np.frombuffer(temp, dtype=np.uint8)
        shuye_area = shuye_area + len(temp)/2
    for bingban_score, bingban in zip(bingban_scores, bingbans):
        if bingban_score[-1] < 0.5:
            continue
        temp = bingban['counts']
        temp = np.frombuffer(temp, dtype=np.uint8)
        bingban_area = bingban_area + len(temp)/2

    print(f'image_id:{i}, 树叶像素:{shuye_area}, 病斑像素:{bingban_area}, 病斑比例: {bingban_area/shuye_area}')
