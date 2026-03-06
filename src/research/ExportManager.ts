/**
 * ExportManager.ts
 * 
 * Generates the research package JSON payload combining benchmark data,
 * hardware telemetry, and dataset metadata.
 */

import type { ComparisonResult } from '../ml/types';
import type { DataCollector } from '../ml/DataCollector';

export function generateResearchJSON(
    comparison: ComparisonResult,
    dataset: DataCollector
): string {
    const counts = dataset.getSampleCounts();
    const sortedLabels = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);

    // Basic hardware/environment telemetry
    const canvas = document.createElement('canvas');
    let gpuInfo = 'Unknown';
    try {
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            const debugInfo = (gl as WebGLRenderingContext).getExtension('WEBGL_debug_renderer_info');
            gpuInfo = debugInfo ? (gl as WebGLRenderingContext).getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) : 'WebGL supported';
        }
    } catch (e) {
        console.warn("Could not retrieve GPU info", e);
    }

    const payload = {
        project: "SignBridge",
        date: new Date().toISOString(),
        dataset: {
            totalSamples: dataset.getTotalSamples(),
            numClasses: Object.keys(counts).length,
            samplesPerClass: counts,
            dominantClass: sortedLabels.length > 0 ? sortedLabels[0] : null,
            minorClass: sortedLabels.length > 0 ? sortedLabels[sortedLabels.length - 1] : null,
        },
        baseline: comparison.baseline,
        custom: comparison.custom,
        comparison: {
            overallMAP: {
                baseline: comparison.baselineMAP,
                custom: comparison.customMAP,
                delta: comparison.customMAP - comparison.baselineMAP
            },
            overallLatency: {
                baseline: comparison.baselineAvgLatency,
                custom: comparison.customAvgLatency,
                delta: comparison.customAvgLatency - comparison.baselineAvgLatency
            },
            modelSizeMB: {
                baseline: comparison.baselineModelSizeMB,
                custom: comparison.customModelSizeMB
            }
        },
        hardware: {
            userAgent: navigator.userAgent,
            gpu: gpuInfo,
            logicalProcessors: navigator.hardwareConcurrency || 'Unknown',
            memoryTotalGB: (navigator as any).deviceMemory || 'Unknown'
        }
    };

    return JSON.stringify(payload, null, 2);
}
