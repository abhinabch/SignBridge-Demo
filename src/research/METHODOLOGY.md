# SignBridge — Research Methodology

## 1. Model Architectures

### Static Gesture Classifier (GestureClassifier.ts)

**Type:** TensorFlow.js Sequential Model

| Layer | Configuration |
|-------|--------------|
| Input | 63 features (21 landmarks × 3 coordinates, flattened) |
| Dense | 128 units, ReLU activation, L2 regularization (λ=0.001) |
| Dropout | Rate: 0.3 |
| Dense | 64 units, ReLU activation |
| Dropout | Rate: 0.2 |
| Output | Softmax, units = number of gesture classes |

**Loss:** Categorical Cross-Entropy  
**Optimizer:** Adam  
**Metrics:** Accuracy

### Sequence Classifier (SequenceClassifier.ts)

**Type:** TensorFlow.js LSTM-based Sequential Model

| Layer | Configuration |
|-------|--------------|
| Input | [N frames, 63 features] (N=20 default) |
| TimeDistributed Dense | 32 units, ReLU activation |
| LSTM | 64 units |
| Dense | 32 units, ReLU activation |
| Output | Softmax, units = number of dynamic gesture classes |

### Classification Routing Strategy

A motion-threshold router determines which classifier to invoke:
- **Motion magnitude < 8px average wrist displacement** → Static classifier
- **Motion magnitude ≥ 8px** → Sequence (temporal) classifier

## 2. Training Parameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Epochs | 50 |
| Batch size | 16 |
| Validation split | 20% |
| Shuffle | Enabled |
| Optimizer | Adam |

## 3. Dataset Collection Protocol

1. **Recording Setup:** User positions hand in front of webcam at ~2ft distance.
2. **Sample Capture:** Each "Record" press captures ~30 frames (~1 second at 30fps).
3. **Per-Frame Data:** 21 hand landmarks (MediaPipe Handpose), each with [x, y, z] coordinates.
4. **Labeling:** Each recording session is tagged with the target gesture label.
5. **Minimum Samples:** Recommended ≥30 samples per gesture class for initial training.
6. **Data Format:** JSON array of `{ label, landmarks[][], timestamp }` objects.
7. **Export:** Dataset can be downloaded as JSON for reproducibility and sharing.

### Supported Gestures (41 total)

- **Alphabet:** A–Z (26 letters)
- **Common Signs:** Hello, Thank You, Yes, No, Please, Sorry, Help, More, Stop, Good, Bad, Water, Eat, Home, Love (15 signs)

## 4. Evaluation Methodology

### Benchmark Runner

The `BenchmarkRunner` evaluates classifier performance by:

1. Running each test sample through the classifier N times (default N=3).
2. Computing per-gesture metrics:
   - **True Positives (TP):** Correct predictions for this gesture.
   - **False Positives (FP):** Other gestures incorrectly predicted as this gesture.
   - **False Negatives (FN):** This gesture incorrectly predicted as another.
   - **Precision:** TP / (TP + FP)
   - **Recall:** TP / (TP + FN)
   - **F1 Score:** 2 × (Precision × Recall) / (Precision + Recall)
   - **Average Latency:** Mean inference time per sample.
3. Computing aggregate metrics:
   - **Overall Accuracy:** Total correct / total predictions.
   - **Mean Average Precision (mAP):** Average of per-class precision scores.
4. Generating a confusion matrix (actual × predicted).

### Hand Tracking Pipeline

- **Model:** MediaPipe Handpose via `@tensorflow-models/handpose`
- **Max Hands:** 2 (two-handed sign support)
- **Handedness Detection:** X-centroid sorting (leftmost in mirrored view = right hand)
- **Feature Vector:** 63 features (single hand) or 126 features (two hands concatenated)

### Real-Time Metrics (Research Overlay)

Toggle with `Shift+D` during live demo:
- FPS of landmark detection pipeline
- Inference latency (10-frame rolling average)
- Active classifier (static / temporal)
- Raw confidence score (0.00–1.00)
- Top 3 candidate gestures with confidence bars
- Active hand count and handedness labels
