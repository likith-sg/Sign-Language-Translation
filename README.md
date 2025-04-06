# Sign Language Translation

## Introduction
This project focuses on translating sign language gestures into text using deep learning and computer vision. By processing video input, the system predicts the corresponding sign language meaning in real-time, making communication more accessible for individuals with hearing or speech impairments.

## Methodology
The approach involves a deep learning pipeline trained on a dataset of sign language gestures. The process consists of:

1. **Dataset Collection:** Capturing sign images using OpenCV from a webcam feed.
2. **Data Augmentation:** Enhancing dataset quality through transformations such as rotation, flipping, brightness changes, and noise addition.
3. **Model Architecture:** Utilizing a pre-trained MobileNetV3 model fine-tuned with additional custom layers for classification.
4. **Training Process:** Optimized with the Adam optimizer and categorical cross-entropy loss to improve classification accuracy.
5. **Inference:** Running real-time predictions on new sign language gestures using webcam input.

## Implementation
### Creating a Custom Dataset
#### Capture Sign Images
- Run `datacollection.py` to collect sign gesture images using a webcam.
- Press **'S'** to capture an image.
- Press **'Q'** to quit the script.
- Images are automatically stored in labeled folders for each gesture.

#### Apply Data Augmentation
- Run `dataAug.py` to apply transformations to the dataset.
- Augmented images enhance model generalization and prevent overfitting.

### Training the Model
#### Prepare the Dataset
- Ensure that the dataset is properly labeled and structured for training.
- Convert images into NumPy arrays for efficient processing.

#### Train the Deep Learning Model
- Run `model3.py` to train the MobileNetV3-based classifier.
- Uses Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss function.

**Training Parameters:**
- Batch size: 32
- Epochs: 50
- Validation Split: 20%
- Model checkpoints are saved for resuming training if interrupted.

### Running Inference
#### Load the Trained Model
- Ensure `mobilenetv3_sign_language_model.keras` is available in the working directory.
- Run `test.py` to classify sign gestures in real-time using webcam input.
- Press **'Q'** to quit the script.
- The model processes video frames, detects hands, and predicts the corresponding sign language gesture.

## Model Performance
| Metric              | Value  |
|--------------------|--------|
| Training Accuracy  | 43.53% |
| Training Loss      | 1.4674 |
| Test Accuracy      | 53.52% |
| Test Loss         | 1.2869 |
| Total Training Time | 2825.66 sec |

**Logs are stored at:** `logs/fit/20250303-121854`

## Running the Project
### Prerequisites
- Python 3.10
- TensorFlow 2.10
- OpenCV (for video capture and preprocessing)
- NumPy, Matplotlib (for data handling and visualization)
- MediaPipe (for hand detection and tracking)

### GPU Acceleration (Optional)
Since training and inference are computationally intensive, GPU acceleration can significantly reduce processing time.

1. **Install NVIDIA CUDA Toolkit**
   - Download CUDA 11.8 (compatible with TensorFlow 2.10) from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive).

2. **Install cuDNN**
   - Download cuDNN 8.6.0 (compatible with CUDA 11.8) from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn-download-survey).
   - Extract files and move them to the CUDA installation directory.

3. **Verify GPU Installation**
   ```sh
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   - If a GPU is detected, the output will list the available GPU devices.

## Steps to Execute the Project
1. **Dataset Collection:**
   ```sh
   python datacollection.py
   ```
   - Press **'S'** to capture images.
   - Press **'Q'** to quit the script.

2. **Data Augmentation:**
   ```sh
   python dataAug.py
   ```

3. **Train the Model:**
   ```sh
   python model3.py
   ```

4. **Run Inference (Real-time Sign Prediction):**
   ```sh
   python test.py
   ```
   - Press **'Q'** to quit the script.

## License
This project is licensed under the MIT License.
