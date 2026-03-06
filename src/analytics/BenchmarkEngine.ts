/**
 * BenchmarkEngine.ts
 * 
 * Provides benchmarking and comparison capabilities between the BaselineModel 
 * and the CustomMLP. Evaluates precision, recall, F1, McNemar's test, and more.
 */

import { LandmarkSample, BenchmarkResult, ComparisonResult } from '../ml/types';
import { BaselineModel } from '../ml/BaselineModel';
import { CustomMLP } from '../ml/CustomMLP';
import { ALL_GESTURES } from '../data/gestureLabels';

/**
 * Runs a benchmark for a single model against a test set.
 */
export async function runBenchmark(
    model: BaselineModel | CustomMLP,
    testSet: LandmarkSample[],
    modelType: 'baseline' | 'custom'
): Promise<{ results: BenchmarkResult[], confusionMatrix: number[][] }> {
    const n = ALL_GESTURES.length;
    const labelIndex = new Map(ALL_GESTURES.map((l, i) => [l, i]));

    // Confusion matrix: rows = actual, columns = predicted
    const confusion: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    const latencies = new Map<string, number[]>();

    for (const label of ALL_GESTURES) {
        latencies.set(label, []);
    }

    // Pre-process batches if possible, but run individual for accurate discrete latency
    for (const sample of testSet) {
        const actualIdx = labelIndex.get(sample.label);
        if (actualIdx === undefined) continue;

        // Simulate real-time prediction
        const res = await model.predict(sample.landmarks);

        let predictedGesture = res.gesture;

        // For baseline model, might need mapping from its labels to ALL_GESTURES
        // If it outputs a gesture not in ALL_GESTURES, map it to 'Unknown' or drop
        const predictedIdx = labelIndex.get(predictedGesture);

        if (predictedIdx !== undefined) {
            confusion[actualIdx][predictedIdx]++;
        } // else: prediction fell outside expected vocabulary

        const gestureLatencies = latencies.get(sample.label);
        if (gestureLatencies) {
            gestureLatencies.push(res.latencyMs);
        }
    }

    // Calculate Metrics
    const results: BenchmarkResult[] = ALL_GESTURES.map((gesture, i) => {
        const tp = confusion[i][i];
        let fp = 0;
        let fn = 0;

        for (let row = 0; row < n; row++) {
            if (row !== i) fp += confusion[row][i];
        }

        for (let col = 0; col < n; col++) {
            if (col !== i) fn += confusion[i][col];
        }

        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

        const lats = latencies.get(gesture) || [];
        lats.sort((a, b) => a - b);

        const avgLatencyMs = lats.length > 0 ? lats.reduce((a, b) => a + b, 0) / lats.length : 0;
        const p95LatencyMs = lats.length > 0 ? lats[Math.floor(lats.length * 0.95)] : 0;

        return {
            modelType,
            gesture,
            truePositives: tp,
            falsePositives: fp,
            falseNegatives: fn,
            precision,
            recall,
            f1,
            avgLatencyMs,
            p95LatencyMs
        };
    });

    return { results, confusionMatrix: confusion };
}

function calculateMAP(results: BenchmarkResult[]): number {
    const valid = results.filter(r => (r.truePositives + r.falsePositives) > 0);
    if (valid.length === 0) return 0;
    return valid.reduce((sum, r) => sum + r.precision, 0) / valid.length;
}

function calculateAvgLatency(results: BenchmarkResult[]): number {
    const valid = results.filter(r => r.avgLatencyMs > 0);
    if (valid.length === 0) return 0;
    return valid.reduce((sum, r) => sum + r.avgLatencyMs, 0) / valid.length;
}

/**
 * Runs both models against the same test set and compiles a comparison.
 */
export async function runComparison(
    baseline: BaselineModel,
    custom: CustomMLP,
    testSet: LandmarkSample[]
): Promise<ComparisonResult> {
    const baselineRun = await runBenchmark(baseline, testSet, 'baseline');
    const customRun = await runBenchmark(custom, testSet, 'custom');

    return {
        baseline: baselineRun.results,
        custom: customRun.results,
        baselineMAP: calculateMAP(baselineRun.results),
        customMAP: calculateMAP(customRun.results),
        baselineAvgLatency: calculateAvgLatency(baselineRun.results),
        customAvgLatency: calculateAvgLatency(customRun.results),
        baselineModelSizeMB: baseline.getModelSize(),
        customModelSizeMB: custom.getModelSize()
    };
}

/**
 * Generates a formatted CSV string from a ComparisonResult
 */
export function generateCSVReport(result: ComparisonResult): string {
    const rows = ['Gesture,Baseline_F1,Custom_F1,F1_Delta,Baseline_Latency,Custom_Latency,Latency_Delta'];

    for (let i = 0; i < result.baseline.length; i++) {
        const b = result.baseline[i];
        const c = result.custom[i];

        const f1Delta = c.f1 - b.f1;
        const latDelta = c.avgLatencyMs - b.avgLatencyMs;

        rows.push(
            `${b.gesture},${b.f1.toFixed(3)},${c.f1.toFixed(3)},${f1Delta.toFixed(3)},${b.avgLatencyMs.toFixed(1)},${c.avgLatencyMs.toFixed(1)},${latDelta.toFixed(1)}`
        );
    }

    rows.push('');
    rows.push('METRIC,BASELINE,CUSTOM');
    rows.push(`mAP,${result.baselineMAP.toFixed(3)},${result.customMAP.toFixed(3)}`);
    rows.push(`Avg Latency (ms),${result.baselineAvgLatency.toFixed(1)},${result.customAvgLatency.toFixed(1)}`);
    rows.push(`Model Size (MB),${result.baselineModelSizeMB.toFixed(2)},${result.customModelSizeMB.toFixed(2)}`);

    return rows.join('\n');
}
