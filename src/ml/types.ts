/**
 * types.ts
 * 
 * Shared interfaces and types for the SignBridge machine learning
 * and benchmarking pipeline.
 */

export interface LandmarkSample {
    landmarks: number[][];   // [21, 3] usually
    label: string;
    handedness: 'Left' | 'Right' | 'Both';
    timestamp: number;
}

export interface PredictionResult {
    gesture: string;
    confidence: number;
    latencyMs: number;
    modelType: 'baseline' | 'custom';
}

export interface BenchmarkResult {
    modelType: 'baseline' | 'custom';
    gesture: string;
    truePositives: number;
    falsePositives: number;
    falseNegatives: number;
    precision: number;
    recall: number;
    f1: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
}

export interface ComparisonResult {
    baseline: BenchmarkResult[];
    custom: BenchmarkResult[];
    baselineMAP: number;
    customMAP: number;
    baselineAvgLatency: number;
    customAvgLatency: number;
    baselineModelSizeMB: number;
    customModelSizeMB: number;
}

export interface TrainOptions {
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    validationSplit?: number;
    callbacks?: any; // tf.CustomCallbackArgs
}

export interface BatchResult {
    sample: LandmarkSample;
    prediction: PredictionResult;
}
