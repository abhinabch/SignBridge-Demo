/**
 * CustomMLP.ts
 * 
 * Trainable TensorFlow.js Sequential model for ASL gesture classification.
 * Replaces the heuristic rules with a neural network trained directly in the browser.
 * 
 * Architecture:
 *   Input:   63 features (21 landmarks × 3 coordinates, flattened)
 *   Dense:   256 units, ReLU
 *   BatchNorm
 *   Dropout: 0.3
 *   Dense:   128 units, ReLU
 *   BatchNorm
 *   Dropout: 0.2
 *   Dense:   64 units, ReLU
 *   Dropout: 0.1
 *   Output:  softmax, units = number of gesture classes
 */

import * as tf from '@tensorflow/tfjs';
import { LandmarkSample, PredictionResult, TrainOptions, BatchResult } from './types';

export interface TrainingProgress {
    epoch: number;
    loss: number;
    valLoss?: number;
    accuracy: number;
    valAccuracy?: number;
}

// Flatten 21×3 landmarks into a 63-element 1D array
export function flattenLandmarks(landmarks: number[][]): number[] {
    const flat: number[] = [];
    for (let i = 0; i < 21; i++) {
        const lm = landmarks[i] || [0, 0, 0];
        flat.push(lm[0], lm[1], lm[2] ?? 0);
    }
    return flat;
}

export class CustomMLP {
    private model: tf.Sequential | null = null;
    private labels: string[] = [];

    /**
     * Builds the MLP architecture.
     */
    build(numClasses: number): void {
        this.model = tf.sequential();

        this.model.add(tf.layers.dense({ inputShape: [63], units: 256, activation: 'relu' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dropout({ rate: 0.3 }));

        this.model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
        this.model.add(tf.layers.batchNormalization());
        this.model.add(tf.layers.dropout({ rate: 0.2 }));

        this.model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        this.model.add(tf.layers.dropout({ rate: 0.1 }));

        this.model.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
    }

    /**
     * Train the model on collected samples with early stopping.
     */
    async train(
        dataset: LandmarkSample[],
        options: TrainOptions = {},
        onProgress?: (progress: TrainingProgress) => void
    ): Promise<TrainingProgress[]> {
        // Determine unique labels
        const labelSet = new Set(dataset.map(s => s.label));
        this.labels = Array.from(labelSet).sort();
        const numClasses = this.labels.length;
        const labelToIndex = new Map(this.labels.map((l, i) => [l, i]));

        // Prepare tensors
        const xs: number[][] = [];
        const ys: number[] = [];

        for (const sample of dataset) {
            xs.push(flattenLandmarks(sample.landmarks));
            ys.push(labelToIndex.get(sample.label)!);
        }

        const xTensor = tf.tensor2d(xs);
        const yTensor = tf.oneHot(tf.tensor1d(ys, 'int32'), numClasses);

        this.build(numClasses);

        if (!this.model) throw new Error("Model build failed");

        this.model.compile({
            optimizer: tf.train.adam(options.learningRate || 0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        const historyLog: TrainingProgress[] = [];

        const earlyStopping = tf.callbacks.earlyStopping({
            monitor: 'val_loss',
            patience: 10,
            mode: 'auto'
        });

        const callbacks = [
            earlyStopping,
            {
                onEpochEnd: async (epoch: number, logs?: tf.Logs) => {
                    if (logs) {
                        const prog: TrainingProgress = {
                            epoch: epoch + 1,
                            loss: logs['loss'] as number,
                            valLoss: logs['val_loss'] as number,
                            accuracy: logs['acc'] as number,
                            valAccuracy: logs['val_acc'] as number,
                        };
                        historyLog.push(prog);
                        if (onProgress) onProgress(prog);
                    }
                }
            }
        ];

        // Combine provided callbacks
        if (options.callbacks) {
            callbacks.push(options.callbacks);
        }

        await this.model.fit(xTensor, yTensor, {
            epochs: options.epochs || 50,
            batchSize: options.batchSize || 16,
            shuffle: true,
            validationSplit: options.validationSplit || 0.2,
            callbacks
        });

        // Cleanup tensors
        xTensor.dispose();
        yTensor.dispose();

        return historyLog;
    }

    /**
     * Export the currently learned labels.
     */
    getLabels(): string[] {
        return this.labels;
    }

    async predict(landmarks: number[][]): Promise<PredictionResult> {
        if (!this.model) throw new Error("CustomMLP is not trained or loaded.");

        const start = performance.now();

        const flat = flattenLandmarks(landmarks);
        const inputTensor = tf.tensor2d([flat]);
        const prediction = this.model.predict(inputTensor) as tf.Tensor;
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

        const latencyMs = performance.now() - start;

        return {
            gesture: this.labels[maxIdx] || 'Unknown',
            confidence: maxConf,
            latencyMs,
            modelType: 'custom'
        };
    }

    async predictBatch(samples: LandmarkSample[]): Promise<BatchResult[]> {
        const results: BatchResult[] = [];
        for (const sample of samples) {
            try {
                const pred = await this.predict(sample.landmarks);
                results.push({ sample, prediction: pred });
            } catch {
                results.push({
                    sample,
                    prediction: { gesture: 'Error', confidence: 0, latencyMs: 0, modelType: 'custom' }
                })
            }
        }
        return results;
    }

    async save(): Promise<void> {
        if (!this.model) return;
        await this.model.save('localstorage://signbridge-custom-mlp');
        localStorage.setItem('signbridge-custom-labels', JSON.stringify(this.labels));
    }

    async load(): Promise<boolean> {
        try {
            const labelsJson = localStorage.getItem('signbridge-custom-labels');
            if (!labelsJson) return false;

            this.model = await tf.loadLayersModel('localstorage://signbridge-custom-mlp') as tf.Sequential;
            this.labels = JSON.parse(labelsJson) as string[];
            return true;
        } catch {
            return false;
        }
    }

    getModelSummary(): { layers: any[], totalParams: number } {
        if (!this.model) return { layers: [], totalParams: 0 };
        return {
            layers: this.model.layers.map(l => ({ name: l.name, type: l.getClassName(), trainableParams: l.countParams() })),
            totalParams: this.model.countParams()
        };
    }

    getModelSize(): number {
        if (!this.model) return 0;
        // 4 bytes per float parameter
        return (this.model.countParams() * 4) / (1024 * 1024);
    }
}
