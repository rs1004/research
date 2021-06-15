from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt = COCO('/home/sato/work/object_detection/data/voc07+12/annotations/instances_val.json')
cocoDt = cocoGt.loadRes('/home/sato/work/research/ssd/pred_val_2.json')
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
