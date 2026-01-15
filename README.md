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
- **Animation**: [Framer Motion](https://www.framer.com/motion/) (implied via design system)

## 🧠 Technical Overview

### Real-Time Hand Tracking
The core of SignBridge is built on the **MediaPipe Handpose** model. The application captures video frames from the user's camera and processes them locally using TensorFlow.js to detect 21 distinct hand landmarks (3D coordinates).

### Gesture Recognition Engine
Recognition is implemented via a heuristic-based engine that analyzes the relationships between landmarks:
- **Finger Extension**: Calculating the relative distance between finger tips and MCP joints.
- **Palm Orientation**: Determining the vector of the hand relative to the camera.
- **Pattern Matching**: Comparing current hand states against pre-defined ASL gesture patterns (e.g., 'Hello', 'Thank You', 'Yes/No').

### Audio Synthesis
Integrated with the **Web Speech API**, the application provides immediate auditory feedback for recognized signs, allowing for a two-way communication flow.

## ✨ Key Features

- **Live Recognition Feed**: AI-powered overlay showing tracked hand points and real-time translation.
- **Demo Mode**: A simulated environment for testing features without requiring a physical camera.
- **Learn Section**: Interactive guides for users to practice and expand their sign language vocabulary.
- **History & Analytics**: Tracking of recent translations for quick reference.
- **Responsive Mockup**: Housed within a premium mobile viewport design to demonstrate mobile application potential.

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