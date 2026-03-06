/**
 * SequenceClassifier.ts
 * 
 * TensorFlow.js LSTM-based model for recognizing dynamic/temporal ASL signs
 * that require motion (e.g., J, Z, Thank You, Sorry).
 * 
 * Architecture:
 *   TimeDistributed Dense(32) over the frame sequence
 *   LSTM(64)
 *   Dense(32, ReLU)
 *   Softmax output
 */

import * as tf from '@tensorflow/tfjs';
import { TemporalBuffer } from './TemporalBuffer';

export interface SequencePrediction {
    gesture: string;
    confidence: number;
}

/** Dynamic signs that require motion detection */
export const DYNAMIC_SIGNS = [
    'J', 'Z', 'Thank You', 'Sorry', 'More', 'Water', 'Eat', 'Home',
    'Hello', 'Please', 'Help', 'Stop',
];

/**
 * Build the LSTM-based sequence classifier.
 * @param numFrames  Number of frames in the input sequence
 * @param numClasses Number of dynamic gesture classes
 */
export function buildSequenceModel(numFrames: number, numClasses: number): tf.Sequential {
    const model = tf.sequential();

    // TimeDistributed Dense(32) over each frame's 63 features
    model.add(tf.layers.timeDistributed({
        layer: tf.layers.dense({ units: 32, activation: 'relu' }),
        inputShape: [numFrames, 63], // [frames, 21 landmarks × 3 coords]
    }));

    // LSTM layer to capture temporal dependencies
    model.add(tf.layers.lstm({
        units: 64,
        returnSequences: false,
    }));

    // Dense classification head
    model.add(tf.layers.dense({
        units: 32,
        activation: 'relu',
    }));

    model.add(tf.layers.dense({
        units: numClasses,
        activation: 'softmax',
    }));

    return model;
}

/**
 * Flatten a single frame's 21×3 landmarks into a 63-element vector.
 */
function flattenFrame(landmarks: number[][]): number[] {
    const flat: number[] = [];
    for (let i = 0; i < 21; i++) {
        const lm = landmarks[i] || [0, 0, 0];
        flat.push(lm[0], lm[1], lm[2] ?? 0);
    }
    return flat;
}

/**
 * Predict a dynamic gesture from a temporal buffer of frames.
 */
export function predictSequence(
    buffer: TemporalBuffer,
    model: tf.LayersModel,
    labels: string[],
): SequencePrediction {
    const sequence = buffer.getSequence();

    // Flatten each frame into a 63-feature vector
    const flatSeq = sequence.map(flattenFrame);

    // Pad or truncate to expected frame count
    const numFrames = (model.inputs[0].shape as number[])[1] || 20;
    while (flatSeq.length < numFrames) {
        flatSeq.unshift(new Array(63).fill(0)); // pad with zeros at the beginning
    }
    const trimmed = flatSeq.slice(-numFrames);

    const inputTensor = tf.tensor3d([trimmed]); // [1, numFrames, 63]
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const probabilities = prediction.dataSync();
    inputTensor.dispose();
    prediction.dispose();

    let maxIdx = 0;
    let maxConf = 0;
    for (let i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxConf) {
            maxConf = probabilities[i];
            maxIdx = i;
        }
    }

    return {
        gesture: labels[maxIdx] || 'Unknown',
        confidence: maxConf,
    };
}

/**
 * Save the sequence model to localStorage.
 */
export async function saveSequenceModel(model: tf.LayersModel, labels: string[]): Promise<void> {
    await model.save('localstorage://signbridge-sequence-model');
    localStorage.setItem('signbridge-sequence-labels', JSON.stringify(labels));
}

/**
 * Load the sequence model from localStorage.
 */
export async function loadSequenceModel(): Promise<{ model: tf.LayersModel; labels: string[] } | null> {
    try {
        const labelsJson = localStorage.getItem('signbridge-sequence-labels');
        if (!labelsJson) return null;
        const model = await tf.loadLayersModel('localstorage://signbridge-sequence-model');
        const labels = JSON.parse(labelsJson) as string[];
        return { model, labels };
    } catch {
        return null;
    }
}
