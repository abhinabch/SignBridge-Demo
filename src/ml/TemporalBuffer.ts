/**
 * TemporalBuffer.ts
 * 
 * A circular buffer that stores the last N frames of landmark data
 * for temporal/dynamic gesture recognition. Used to detect motion-based
 * signs (e.g., J, Z, Thank You) that require analyzing movement over time.
 */

export class TemporalBuffer {
    private buffer: number[][][] = [];
    private maxSize: number;
    private pointer: number = 0;
    private filled: boolean = false;

    /**
     * @param maxSize Number of frames to buffer (default: 20)
     */
    constructor(maxSize: number = 20) {
        this.maxSize = maxSize;
        this.buffer = new Array(maxSize).fill(null);
    }

    /**
     * Push a frame of landmarks (21×3) into the circular buffer.
     */
    push(landmarks: number[][]): void {
        this.buffer[this.pointer] = landmarks;
        this.pointer = (this.pointer + 1) % this.maxSize;
        if (this.pointer === 0) {
            this.filled = true;
        }
    }

    /**
     * Returns the full buffer as a 3D array [frames, 21, 3],
     * ordered oldest→newest.
     */
    getSequence(): number[][][] {
        if (!this.filled) {
            // Return only filled portion
            return this.buffer.slice(0, this.pointer).filter(Boolean);
        }
        // Return in chronological order: from pointer to end, then start to pointer
        const ordered = [
            ...this.buffer.slice(this.pointer),
            ...this.buffer.slice(0, this.pointer),
        ];
        return ordered.filter(Boolean);
    }

    /**
     * Whether the buffer has been completely filled at least once.
     */
    isFull(): boolean {
        return this.filled;
    }

    /**
     * Returns the current number of valid frames in the buffer.
     */
    getLength(): number {
        return this.filled ? this.maxSize : this.pointer;
    }

    /**
     * Calculates the average wrist (landmark 0) displacement across
     * the buffer. Used to detect whether the hand is moving (dynamic sign)
     * or stationary (static sign).
     * 
     * Returns the magnitude of the average displacement vector.
     */
    getMotionVector(): number {
        const sequence = this.getSequence();
        if (sequence.length < 2) return 0;

        let totalDx = 0;
        let totalDy = 0;
        let totalDz = 0;
        let count = 0;

        for (let i = 1; i < sequence.length; i++) {
            const prev = sequence[i - 1];
            const curr = sequence[i];

            if (!prev || !curr || prev.length < 1 || curr.length < 1) continue;

            // Wrist is landmark 0
            const dx = curr[0][0] - prev[0][0];
            const dy = curr[0][1] - prev[0][1];
            const dz = (curr[0][2] ?? 0) - (prev[0][2] ?? 0);

            totalDx += dx;
            totalDy += dy;
            totalDz += dz;
            count++;
        }

        if (count === 0) return 0;

        const avgDx = totalDx / count;
        const avgDy = totalDy / count;
        const avgDz = totalDz / count;

        return Math.sqrt(avgDx * avgDx + avgDy * avgDy + avgDz * avgDz);
    }

    /**
     * Reset the buffer.
     */
    clear(): void {
        this.buffer = new Array(this.maxSize).fill(null);
        this.pointer = 0;
        this.filled = false;
    }
}
