import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Database, Play, Square, Download, Brain, Loader2, CheckCircle } from 'lucide-react';
import { DataCollector } from '../ml/DataCollector';
import { CustomMLP, type TrainingProgress } from '../ml/CustomMLP';
import type { LandmarkSample } from '../ml/types';

// All supported gesture labels
const ALPHABET_SIGNS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const COMMON_SIGNS = [
    'Hello', 'Thank You', 'Yes', 'No', 'Please',
    'Sorry', 'Help', 'More', 'Stop', 'Good',
    'Bad', 'Water', 'Eat', 'Home', 'Love',
];
const ALL_LABELS = [...ALPHABET_SIGNS, ...COMMON_SIGNS];

type ModelStatus = 'untrained' | 'training' | 'ready';

interface TrainingPanelProps {
    /** Called when model finishes training — parent can use the model for inference */
    onModelReady?: (model: CustomMLP) => void;
    /** External landmarks feed — called each frame when recording */
    currentLandmarks?: number[][] | null;
}

export function TrainingPanel({ onModelReady, currentLandmarks }: TrainingPanelProps) {
    const [modelStatus, setModelStatus] = useState<ModelStatus>('untrained');
    const [trainingHistory, setTrainingHistory] = useState<TrainingProgress[]>([]);
    const [recordingLabel, setRecordingLabel] = useState<string | null>(null);
    const [sampleCounts, setSampleCounts] = useState<Map<string, number>>(new Map());
    const [frameCount, setFrameCount] = useState(0);
    const collectorRef = useRef(new DataCollector());
    const recordingTimerRef = useRef<number | null>(null);
    const frameCounterRef = useRef(0);

    const mlpRef = useRef(new CustomMLP());

    // Try to load a previously saved model on mount
    useEffect(() => {
        (async () => {
            const success = await mlpRef.current.load();
            if (success) {
                setModelStatus('ready');
                onModelReady?.(mlpRef.current);
            }
        })();
    }, []);

    // Feed frames into collector when recording
    useEffect(() => {
        if (recordingLabel && currentLandmarks && currentLandmarks.length >= 21) {
            collectorRef.current.addFrame(currentLandmarks);
            frameCounterRef.current += 1;
            setFrameCount(frameCounterRef.current);
        }
    }, [currentLandmarks, recordingLabel]);

    const startRecording = useCallback((label: string) => {
        collectorRef.current.startRecording(label);
        setRecordingLabel(label);
        frameCounterRef.current = 0;
        setFrameCount(0);

        // Auto-stop after ~30 frames (at ~30fps → ~1 second)
        recordingTimerRef.current = window.setTimeout(() => {
            stopRecording();
        }, 2000);
    }, []);

    const stopRecording = useCallback(() => {
        collectorRef.current.stopRecording();
        setRecordingLabel(null);
        setSampleCounts(new Map(Object.entries(collectorRef.current.getSampleCounts())));
        setFrameCount(0);

        if (recordingTimerRef.current) {
            clearTimeout(recordingTimerRef.current);
            recordingTimerRef.current = null;
        }
    }, []);

    const handleTrain = useCallback(async () => {
        const samples = collectorRef.current.getDataset();
        if (samples.length === 0) return;

        setModelStatus('training');
        setTrainingHistory([]);

        try {
            await mlpRef.current.train(
                samples,
                { epochs: 50, batchSize: 16, learningRate: 0.001 },
                (progress) => {
                    setTrainingHistory((prev) => [...prev, progress]);
                }
            );

            await mlpRef.current.save();
            setModelStatus('ready');
            onModelReady?.(mlpRef.current);
        } catch (err) {
            console.error('Training failed:', err);
            setModelStatus('untrained');
        }
    }, [onModelReady]);

    const handleExport = useCallback(() => {
        collectorRef.current.exportJSON();
    }, []);

    const totalSamples = collectorRef.current.getTotalSamples();

    const statusBadge = {
        untrained: <Badge variant="secondary">Untrained</Badge>,
        training: <Badge className="bg-yellow-500 text-white animate-pulse">Training…</Badge>,
        ready: <Badge className="bg-green-600 text-white"><CheckCircle className="w-3 h-3 mr-1 inline" />Ready</Badge>,
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Brain className="w-5 h-5 text-purple-600" />
                            Training Panel
                        </div>
                        {statusBadge[modelStatus]}
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center justify-between gap-4 flex-wrap">
                        <div className="text-sm text-gray-600">
                            Total samples: <strong>{totalSamples}</strong>
                        </div>
                        <div className="flex gap-2">
                            <Button
                                onClick={handleTrain}
                                disabled={totalSamples === 0 || modelStatus === 'training'}
                                size="sm"
                            >
                                {modelStatus === 'training' ? (
                                    <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Training…</>
                                ) : (
                                    <><Brain className="w-4 h-4 mr-2" />Train Model</>
                                )}
                            </Button>
                            <Button onClick={handleExport} variant="outline" size="sm" disabled={totalSamples === 0}>
                                <Download className="w-4 h-4 mr-2" />
                                Export Dataset
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Training Chart */}
            {trainingHistory.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">Training Progress</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={trainingHistory}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="epoch" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Line type="monotone" dataKey="loss" stroke="#ef4444" name="Loss" dot={false} />
                                <Line type="monotone" dataKey="accuracy" stroke="#22c55e" name="Accuracy" dot={false} />
                            </LineChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            )}

            {/* Gesture Labels Grid */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Database className="w-5 h-5 text-blue-600" />
                        Data Collection
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-600 mb-4">
                        Hold the "Record" button for each gesture while performing the sign in front of the camera.
                        Collect at least 30 samples per gesture for best results.
                    </p>
                    <div className="space-y-6">
                        {/* Common Signs */}
                        <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Common Signs</h4>
                            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                                {COMMON_SIGNS.map((label) => {
                                    const count = sampleCounts.get(label) || 0;
                                    const isRecordingThis = recordingLabel === label;
                                    return (
                                        <div key={label} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg border">
                                            <div className="flex-1 min-w-0">
                                                <div className="text-sm font-medium truncate">{label}</div>
                                                <div className="text-xs text-gray-500">{count} samples</div>
                                            </div>
                                            <Button
                                                size="sm"
                                                variant={isRecordingThis ? 'destructive' : 'outline'}
                                                onClick={() => isRecordingThis ? stopRecording() : startRecording(label)}
                                                disabled={modelStatus === 'training' || (!!recordingLabel && !isRecordingThis)}
                                                className="ml-2"
                                            >
                                                {isRecordingThis ? (
                                                    <><Square className="w-3 h-3 mr-1" />{frameCount}</>
                                                ) : (
                                                    <><Play className="w-3 h-3 mr-1" />Rec</>
                                                )}
                                            </Button>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Alphabet */}
                        <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Alphabet (A–Z)</h4>
                            <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-9 gap-2">
                                {ALPHABET_SIGNS.map((label) => {
                                    const count = sampleCounts.get(label) || 0;
                                    const isRecordingThis = recordingLabel === label;
                                    return (
                                        <div key={label} className="flex flex-col items-center p-2 bg-gray-50 rounded-lg border text-center">
                                            <div className="text-sm font-bold">{label}</div>
                                            <div className="text-xs text-gray-500">{count}</div>
                                            <Button
                                                size="sm"
                                                variant={isRecordingThis ? 'destructive' : 'outline'}
                                                onClick={() => isRecordingThis ? stopRecording() : startRecording(label)}
                                                disabled={modelStatus === 'training' || (!!recordingLabel && !isRecordingThis)}
                                                className="mt-1 h-6 text-xs px-2"
                                            >
                                                {isRecordingThis ? <Square className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                                            </Button>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
