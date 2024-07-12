import os
import json
import cv2
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine.config import Config
from mmdet.apis import DetInferencer
from segment_anything import sam_model_registry, SamPredictor

# DEVICE selection
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Paths
MM_GROUNDING_DINO_CONFIG_PATH = "../mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py"
MM_GROUNDING_DINO_CHECKPOINT_PATH = "./mmdetection/checkpoints/grounding_dino_swin-l_pretrain_all-56d69e78.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./Grounded-Segment-Anything/checkpoints/sam_vit_h_4b8939.pth"
COCO_ANNOTATIONS_PATH = "Data/SWL/annotations/SWL/SWL_coco2017.json"
SOURCE_DIR_PATH = "Data/SWL/images"
OUTPUT_DIR_PATH = "mm-grounded-sam/eval"
TEXT_PROMPT = "wall . ceiling . light . speaker . door . smoke alarm . floor . trash bin . elevator button . escape sign . board . fire extinguisher . door sign . light switch . emergency switch button . elevator . handrail . show window . pipes . staircase . window . radiator . socket ."

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8

# Building MM GroundingDINO inference model
config = Config.fromfile(MM_GROUNDING_DINO_CONFIG_PATH)
inferencer = DetInferencer(
    model=MM_GROUNDING_DINO_CONFIG_PATH,
    weights=MM_GROUNDING_DINO_CHECKPOINT_PATH,
    device=DEVICE,
)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Function to process a single image
def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Perform inference with text prompts
    results = inferencer(
        inputs=image_path,
        texts=TEXT_PROMPT,
        return_datasamples=True,
        custom_entities=True)

    # Extracting detections from results
    data_samples = results['predictions'][0]
    detections = data_samples.pred_instances

    # Extract bboxes, scores, and labels from the detections
    bboxes = detections.bboxes.cpu().numpy()
    scores = detections.scores.cpu().numpy()
    labels = detections.labels.cpu().numpy()

    # Apply score threshold
    mask_scores = scores > BOX_THRESHOLD
    bboxes = bboxes[mask_scores]
    scores = scores[mask_scores]
    labels = labels[mask_scores]

    # NMS post-process
    nms_idx = torchvision.ops.nms(
        torch.tensor(bboxes),
        torch.tensor(scores),
        NMS_THRESHOLD
    ).numpy().tolist()

    bboxes = bboxes[nms_idx]
    scores = scores[nms_idx]
    labels = labels[nms_idx]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    # Convert detections to masks
    detections_mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=bboxes
    )

    return bboxes, detections_mask, labels, scores

# Load COCO annotations
coco = COCO(COCO_ANNOTATIONS_PATH)
image_ids = coco.getImgIds()

# Prepare output directory
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Prepare detection results for COCO evaluation
detections = []

for image_id in image_ids:
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(SOURCE_DIR_PATH, image_info['file_name'])

    bboxes, masks, labels, scores = process_image(image_path)

    # Format results for COCO
    for bbox, mask, label, score in zip(bboxes, masks, labels, scores):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        detection = {
            "image_id": image_id,
            "category_id": int(label) + 1,  # COCO categories start from 1
            "bbox": [x1, y1, width, height],
            "score": float(score),
            "segmentation": mask.tolist()
        }
        detections.append(detection)

# Save detection results to a JSON file
detections_output_path = os.path.join(OUTPUT_DIR_PATH, 'detections.json')
with open(detections_output_path, 'w') as f:
    json.dump(detections, f)

# Load detection results for evaluation
coco_dets = coco.loadRes(detections_output_path)

# Evaluate detections using COCO API
coco_eval = COCOeval(coco, coco_dets, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Save evaluation results
eval_output_path = os.path.join(OUTPUT_DIR_PATH, 'evaluation_results.json')
with open(eval_output_path, 'w') as f:
    json.dump(coco_eval.eval, f)
