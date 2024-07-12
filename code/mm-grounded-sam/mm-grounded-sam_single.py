import cv2
import numpy as np
import torch


from mmengine.config import Config
from mmdet.apis import DetInferencer
from segment_anything import sam_model_registry, SamPredictor
import torchvision
from mmdet.visualization.local_visualizer import DetLocalVisualizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MM Detection config and checkpoint
MM_GROUNDING_DINO_CONFIG_PATH = "../mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py"
MM_GROUNDING_DINO_CHECKPOINT_PATH = "./mmdetection/checkpoints/grounding_dino_swin-l_pretrain_all-56d69e78.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./Grounded-Segment-Anything/checkpoints/sam_vit_h_4b8939.pth"

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

# Predict classes and hyper-param for MM GroundingDINO
SOURCE_IMAGE_PATH = "Data/Office_samples/low-res-sample/rgb_12.png"
TEXT_PROMPT = "bar stool"
BOX_THRESHOLD = 0.5
TEXT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8

# Load image
image = cv2.imread(SOURCE_IMAGE_PATH)

# Perform inference with text prompts
results = inferencer(inputs=SOURCE_IMAGE_PATH, texts=TEXT_PROMPT)

# Extracting detections from results
detections = results['predictions'][0]

# Convert detections to the required format
bboxes = np.array(detections['bboxes'])
scores = np.array(detections['scores'])
# Convert detections to the required format
mask_scores = scores > BOX_THRESHOLD
bboxes = bboxes[mask_scores]
scores = scores[mask_scores]
labels = [TEXT_PROMPT] * len(bboxes)

print (len(scores))

# NMS post process
print(f"Before NMS: {len(bboxes)} boxes")
nms_idx = torchvision.ops.nms(
    torch.tensor(bboxes),
    torch.tensor(scores),
    NMS_THRESHOLD
).numpy().tolist()

bboxes = bboxes[nms_idx]
scores = scores[nms_idx]

print(f"After NMS: {len(bboxes)} boxes")

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

# Create an instance of DetLocalVisualizer
visualizer = DetLocalVisualizer()

# Create a DetDataSample object
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

gt_instances = InstanceData()
gt_instances.bboxes = torch.Tensor(bboxes)
gt_instances.labels = torch.Tensor([0] * len(bboxes)).long()
gt_instances.masks = torch.Tensor(detections_mask)

gt_det_data_sample = DetDataSample()
gt_det_data_sample.gt_instances = gt_instances

# Annotate image with detections using DetLocalVisualizer
visualizer.dataset_meta = {'classes': [TEXT_PROMPT], 'palette': [(0, 255, 0)]}
visualizer.add_datasample('image', image, gt_det_data_sample, show=False, out_file='mm_groundingdino_annotated_image.jpg')
