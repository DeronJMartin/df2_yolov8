from ultralytics import YOLO
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available; using CPU.")

# Initialize the YOLOv8 segmentation model without specifying device here
model = YOLO("yolov8n-seg.pt")

# Move the underlying model to the desired device (MPS in this case)
model = model.to(device)

# Train the model with logging and saving enabled
results = model.train(
    data="deepfashion2.yaml",
    epochs=100,
    imgsz=416,
    project="runs/df2_seg",   # Base directory for logging & saving checkpoints
    name="df2_1",             # Name of this experiment/run
    save=True,                # Ensure checkpoints are saved (default True)
    verbose=True              # Print detailed logs during training
)
