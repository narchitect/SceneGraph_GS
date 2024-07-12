import json
import cv2
import numpy as np
import torch
import torchvision
from mmengine.config import Config
from mmdet.apis import DetInferencer
from segment_anything import sam_model_registry, SamPredictor
import os

# DEVICE select
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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

# Set the chunked size for texts
#inferencer.model.test_cfg.chunked_size = 10

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for MM GroundingDINO
SOURCE_DIR_PATH = "Data/SWL/images"
OUTPUT_DIR_PATH = ("mm-grounded-sam/output/SWLtest")
TEXT_PROMPT = "wall . ceiling . light . speaker . door . smoke alarm . floor . trash bin . elevator button . escape sign . board . fire extinguisher . door sign . light switch . emergency switch button . elevator . handrail . show window . pipes . staircase . window . radiator . socket ."

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8

undetected_images = []

# Load image
class_names = TEXT_PROMPT.split(' . ')
num_classes = len(class_names)

# Function to process a single image
def process_image(image_path, output_dir):
    # Load image
    image = cv2.imread(image_path)

    # Perform inference with text prompts
    results = inferencer(
        inputs=image_path,
        texts=TEXT_PROMPT,
        return_datasamples= True,
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

    # If it doesn't detect anything in the image, store the image name on txt file
    if len(bboxes) == 0:
        undetected_images.append(os.path.basename(image_path))
        return

    # NMS post-process
    print(f"Before NMS: {len(bboxes)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.tensor(bboxes),
        torch.tensor(scores),
        NMS_THRESHOLD
    ).numpy().tolist()

    bboxes = bboxes[nms_idx]
    scores = scores[nms_idx]
    labels = labels[nms_idx]

    # Filter detections to match NMS results
    detections = detections[mask_scores]
    detections = detections[nms_idx]

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

    # Add the mask to the existing data samples
    detections.masks = torch.Tensor(detections_mask).to(DEVICE)
    data_samples.pred_instances = detections
    #if there is a ground_truth data on it then it should be added into data sample.

    def save_predictions_as_json(predictions, output_dir, image_name):
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.json")
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)

    # Save predictions as JSON
    prediction_result = inferencer.pred2dict(data_samples)
    save_predictions_as_json(prediction_result, output_dir, os.path.basename(image_path))

    # Get the visualizer from the inferencer
    visualizer = inferencer.visualizer

    # Annotate image with detections using the visualizer from inferencer
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    visualizer.add_datasample('image', image, data_samples, draw_gt = False, draw_pred= True, show=False, out_file=output_path)

    print(f"Processed {os.path.basename(image_path)}")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

# Load images in the input directory
for image_file in os.listdir(SOURCE_DIR_PATH):
    if image_file.endswith(('JPG', 'png', 'jpeg')):
        image_path = os.path.join(SOURCE_DIR_PATH, image_file)
        process_image(image_path, OUTPUT_DIR_PATH)

# Save undetected images in a txt file
with open(os.path.join(OUTPUT_DIR_PATH, 'undetected_images.txt'), 'w') as f:
    for item in undetected_images:
        f.write("%s\n" % item)
