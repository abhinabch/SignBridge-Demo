/**
 * DataCollector.ts
 * 
 * Manages the collection, buffering, and export of labeled landmark data
 * for training the gesture classifier. Samples are stored in-memory and
 * can be exported/imported as JSON.
 */

import { LandmarkSample } from './types';

export class DataCollector {
    private samples: LandmarkSample[] = [];
    private currentLabel: string | null = null;
    private buffer: number[][][] = []; // [frames][21][3]
    private isRecording = false;

    /**
     * Begin buffering landmark frames under the given label.
     */
    startRecording(label: string): void {
        this.currentLabel = label;
        this.buffer = [];
        this.isRecording = true;
    }

    /**
     * Add a frame of landmarks to the active recording buffer.
     * Call this every frame while recording.
     */
    addFrame(landmarks: number[][]): void {
        if (!this.isRecording) return;
        this.buffer.push(landmarks);
    }

    /**
     * Stop recording and save the buffered frames as labeled samples.
     * Each buffered frame becomes a separate training sample.
     * Generates a "Both" handedness as default since the tracker extracts that later.
     */
    stopRecording(): LandmarkSample[] {
        if (!this.isRecording || !this.currentLabel) return [];

        const timestamp = Date.now();
        const newSamples: LandmarkSample[] = [];

        for (const frame of this.buffer) {
            newSamples.push({
                label: this.currentLabel,
                landmarks: frame,
                timestamp,
                handedness: 'Right' // Will be properly tagged by tracker in a combined setup
            });
        }

        this.samples.push(...newSamples);

        this.isRecording = false;
        this.currentLabel = null;
        this.buffer = [];
        return newSamples;
    }

    /**
     * Add multiple samples in bulk
     */
    addSamples(newSamples: LandmarkSample[]): void {
        this.samples.push(...newSamples);
    }

    /**
     * Returns whether the collector is actively recording.
     */
    getIsRecording(): boolean {
        return this.isRecording;
    }

    /**
     * Returns the current recording label, or null if not recording.
     */
    getCurrentLabel(): string | null {
        return this.currentLabel;
    }

    /**
     * Returns the number of frames buffered so far in the current recording.
     */
    getBufferLength(): number {
        return this.buffer.length;
    }

    /**
     * Returns the full dataset
     */
    getDataset(): LandmarkSample[] {
        return [...this.samples];
    }

    /**
     * Trigger a browser download of the dataset as a JSON file.
     */
    exportJSON(): void {
        const dataStr = JSON.stringify(this.samples, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        const a = document.createElement('a');
        a.href = url;
        a.download = `signbridge-dataset-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();

        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Import a dataset from an uploaded JSON file (additive).
     */
    async importJSON(file: File): Promise<void> {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target?.result as string) as LandmarkSample[];
                    if (Array.isArray(data)) {
                        this.addSamples(data);
                        resolve();
                    } else {
                        reject(new Error("Invalid dataset format"));
                    }
                } catch (err) {
                    reject(err);
                }
            };
            reader.onerror = () => reject(new Error("Failed to read file"));
            reader.readAsText(file);
        });
    }

    /**
     * Returns the number of collected samples for a given label.
     */
    getSampleCount(label: string): number {
        return this.samples.filter(s => s.label === label).length;
    }

    /**
     * Returns sample counts for all labels as a Record.
     */
    getSampleCounts(): Record<string, number> {
        const counts: Record<string, number> = {};
        for (const sample of this.samples) {
            counts[sample.label] = (counts[sample.label] || 0) + 1;
        }
        return counts;
    }

    /**
     * Returns total number of collected samples.
     */
    getTotalSamples(): number {
        return this.samples.length;
    }

    /**
     * Removes all samples for a specific label.
     */
    clearLabel(label: string): void {
        this.samples = this.samples.filter(s => s.label !== label);
    }

    /**
     * Clear all collected samples.
     */
    clearAll(): void {
        this.samples = [];
        this.buffer = [];
        this.isRecording = false;
        this.currentLabel = null;
    }
}
