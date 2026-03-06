/**
 * BenchmarkRunner.ts
 * 
 * Evaluates the gesture classifier's performance by running controlled
 * trials against stored test data. Produces per-gesture precision, recall,
 * F1, latency metrics, and overall mean average precision (mAP).
 */

import { ALL_GESTURES } from '../data/gestureLabels';

export interface BenchmarkResult {
    gesture: string;
    truePositives: number;
    falsePositives: number;
    falseNegatives: number;
    precision: number;
    recall: number;
    f1: number;
    avgLatencyMs: number;
}

export interface BenchmarkSummary {
    results: BenchmarkResult[];
    overallAccuracy: number;
    meanAveragePrecision: number;
    avgLatencyMs: number;
    confusionMatrix: number[][];
    labels: string[];
}

export interface TestSample {
    label: string;
    landmarks: number[][];
}

/**
 * Run benchmark trials against a classifier function.
 * 
 * @param classifyFn  Function that takes landmarks and returns { gesture, confidence }
 * @param testSamples Array of labeled test samples
 * @param trials      Number of times to run each sample (for latency averaging)
 */
export function runBenchmark(
    classifyFn: (landmarks: number[][]) => { gesture: string; confidence: number },
    testSamples: TestSample[],
    trials: number = 3,
): BenchmarkSummary {
    const labels = ALL_GESTURES;
    const labelIndex = new Map(labels.map((l, i) => [l, i]));
    const n = labels.length;

    // Confusion matrix: rows = actual, columns = predicted
    const confusion: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

    // Per-gesture latency accumulators
    const latencies = new Map<string, number[]>();
    for (const label of labels) {
        latencies.set(label, []);
    }

    // Run classification trials
    for (const sample of testSamples) {
        const actualIdx = labelIndex.get(sample.label);
        if (actualIdx === undefined) continue;

        for (let t = 0; t < trials; t++) {
            const start = performance.now();
            const result = classifyFn(sample.landmarks);
            const elapsed = performance.now() - start;

            const predictedIdx = labelIndex.get(result.gesture);
            if (predictedIdx !== undefined) {
                confusion[actualIdx][predictedIdx]++;
            }

            const lat = latencies.get(sample.label);
            if (lat) lat.push(elapsed);
        }
    }

    // Compute per-gesture metrics
    const results: BenchmarkResult[] = labels.map((gesture, i) => {
        const tp = confusion[i][i];
        let fp = 0;
        let fn = 0;

        // FP: how many times other gestures were predicted as this gesture
        for (let row = 0; row < n; row++) {
            if (row !== i) fp += confusion[row][i];
        }

        // FN: how many times this gesture was predicted as something else
        for (let col = 0; col < n; col++) {
            if (col !== i) fn += confusion[i][col];
        }

        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

        const gestureLat = latencies.get(gesture) || [];
        const avgLatencyMs = gestureLat.length > 0
            ? gestureLat.reduce((a, b) => a + b, 0) / gestureLat.length
            : 0;

        return {
            gesture,
            truePositives: tp,
            falsePositives: fp,
            falseNegatives: fn,
            precision: Math.round(precision * 1000) / 1000,
            recall: Math.round(recall * 1000) / 1000,
            f1: Math.round(f1 * 1000) / 1000,
            avgLatencyMs: Math.round(avgLatencyMs * 100) / 100,
        };
    });

    // Overall accuracy
    let totalCorrect = 0;
    let totalSamples = 0;
    for (let i = 0; i < n; i++) {
        totalCorrect += confusion[i][i];
        for (let j = 0; j < n; j++) {
            totalSamples += confusion[i][j];
        }
    }
    const overallAccuracy = totalSamples > 0 ? totalCorrect / totalSamples : 0;

    // Mean average precision (average of per-class precision)
    const precisions = results.filter(r => (r.truePositives + r.falsePositives) > 0).map(r => r.precision);
    const mAP = precisions.length > 0 ? precisions.reduce((a, b) => a + b, 0) / precisions.length : 0;

    // Average latency
    const allLat = results.map(r => r.avgLatencyMs).filter(l => l > 0);
    const avgLatencyMs = allLat.length > 0 ? allLat.reduce((a, b) => a + b, 0) / allLat.length : 0;

    return {
        results,
        overallAccuracy: Math.round(overallAccuracy * 1000) / 1000,
        meanAveragePrecision: Math.round(mAP * 1000) / 1000,
        avgLatencyMs: Math.round(avgLatencyMs * 100) / 100,
        confusionMatrix: confusion,
        labels,
    };
}

/**
 * Export benchmark results as CSV string.
 */
export function exportBenchmarkCSV(summary: BenchmarkSummary): string {
    const header = 'Gesture,True Positives,False Positives,False Negatives,Precision,Recall,F1,Avg Latency (ms)';
    const rows = summary.results.map(r =>
        `${r.gesture},${r.truePositives},${r.falsePositives},${r.falseNegatives},${r.precision},${r.recall},${r.f1},${r.avgLatencyMs}`
    );
    const footer = `\nOverall Accuracy,${summary.overallAccuracy}\nMean Avg Precision,${summary.meanAveragePrecision}\nAvg Latency (ms),${summary.avgLatencyMs}`;
    return [header, ...rows].join('\n') + footer;
}
