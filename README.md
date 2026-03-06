# SignBridge: AI-Powered Sign Language Translation Demo

> [!IMPORTANT]  
> **Demo Repository**: This is a demonstration repository showcasing the UI/UX and core technical capabilities of the SignBridge application. The current version is actively being updated.

SignBridge is a modern web application designed to break communication barriers by providing real-time American Sign Language (ASL) translation. Utilizing computer vision and machine learning, it translates hand gestures into text and speech, facilitating seamless interactions between the deaf and hearing communities.

## 🚀 Technical Stack

- **Frontend**: [React](https://reactjs.org/) (v18.3) with [TypeScript](https://www.typescriptlang.org/)
- **Build Tool**: [Vite](https://vitejs.dev/) (v6.3)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **UI Components**: [Radix UI](https://www.radix-ui.com/) & [Lucide React](https://lucide.dev/)
- **AI/ML**: [TensorFlow.js](https://www.tensorflow.org/js) & [@tensorflow-models/handpose](https://github.com/tensorflow/tfjs-models/tree/master/handpose)
- **Charts**: [Recharts](https://recharts.org/) for training visualization and benchmarking

## 🧠 Research

### Trainable Gesture Classifier

The core recognition engine has been upgraded from hardcoded heuristic rules to a **trainable TF.js neural network** with the following architecture:

| Layer | Config |
|-------|--------|
| Input | 63 features (21 landmarks × 3 coords) |
| Dense | 128 units, ReLU, L2(0.001) |
| Dropout | 0.3 |
| Dense | 64 units, ReLU |
| Dropout | 0.2 |
| Output | Softmax (N classes) |

A secondary **LSTM-based Sequence Classifier** handles dynamic signs requiring motion:

| Layer | Config |
|-------|--------|
| TimeDistributed Dense | 32 units, ReLU |
| LSTM | 64 units |
| Dense | 32 units, ReLU |
| Output | Softmax |

### Motion-Based Routing

A temporal buffer tracks wrist displacement over 20 frames. If motion magnitude is below the threshold, the **static classifier** fires; otherwise, the **temporal classifier** is used.

### Two-Handed Support

The hand tracking pipeline supports **up to 2 hands** simultaneously. Handedness is detected via x-coordinate centroid sorting, and landmarks are drawn in distinct colors with left/right labels.

### Research Metrics Overlay

Press **Shift+D** during a live demo to toggle a real-time overlay showing:
- FPS, inference latency (10-frame rolling avg)
- Active classifier (static/temporal)
- Raw confidence, top-3 candidates with bars
- Active hand count

Full methodology documentation: [`src/research/METHODOLOGY.md`](src/research/METHODOLOGY.md)

## 📊 Benchmarks

| Metric | Value |
|--------|-------|
| Overall Accuracy | _TBD — run with collected data_ |
| Mean Avg Precision | _TBD_ |
| Avg Latency | _TBD_ |

Per-gesture precision/recall/F1 scores and a confusion matrix are available in the **Benchmark Dashboard** accessible from the app's home screen.

## 🏋️ Training Panel

The built-in **Training Panel** allows you to:

1. **Collect data**: Click "Record" for each gesture label while performing the sign. Each press captures ~30 frames.
2. **Train**: Click "Train Model" to train the neural network directly in your browser. A live loss/accuracy chart shows training progress.
3. **Save/Load**: Trained models persist in browser `localStorage` automatically.
4. **Export**: Download the collected dataset as JSON for reproducibility.

Access the Training Panel from the app's home screen → **Train** card.

## ✨ Key Features

- **41 Supported Gestures**: A–Z fingerspelling + 15 common signs
- **Live Recognition Feed**: AI-powered overlay with hand skeleton tracking
- **Two-Hand Detection**: Supports left/right hand with colored overlays
- **Demo Mode**: Simulated environment for testing without a camera
- **Sign Glossary**: Browse all signs with difficulty badges and descriptions
- **Benchmark Dashboard**: Per-gesture precision/recall/F1 + confusion matrix
- **Learn Section**: Interactive sign language learning path
- **Text-to-Speech**: Audio feedback for recognized signs

## 🛠️ Development Setup

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Installation
```bash
npm install
```

### Running the App
```bash
npm run dev
```

### Building for Production
```bash
npm run build
```

---
*SignBridge — Breaking barriers, one sign at a time.*