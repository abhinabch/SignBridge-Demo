import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Download, TrendingUp, Clock, Target, ArrowUpDown } from 'lucide-react';
import { type BenchmarkSummary, exportBenchmarkCSV, runBenchmark } from '../analytics/BenchmarkRunner';
import { ALL_GESTURES } from '../data/gestureLabels';

type SortField = 'gesture' | 'precision' | 'recall' | 'f1' | 'avgLatencyMs';
type SortDir = 'asc' | 'desc';

/**
 * recognizeFn should be the same function used for live recognition.
 * It takes landmarks and returns { gesture, confidence }.
 */
interface BenchmarkDashboardProps {
    recognizeFn?: (landmarks: number[][]) => { gesture: string; confidence: number } | null;
    testSamples?: { label: string; landmarks: number[][] }[];
}

export function BenchmarkDashboard({ recognizeFn, testSamples }: BenchmarkDashboardProps) {
    const [summary, setSummary] = useState<BenchmarkSummary | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [sortField, setSortField] = useState<SortField>('f1');
    const [sortDir, setSortDir] = useState<SortDir>('desc');

    const handleRunBenchmark = async () => {
        if (!recognizeFn || !testSamples || testSamples.length === 0) return;
        setIsRunning(true);

        // Run in a setTimeout to allow UI to update
        setTimeout(() => {
            const result = runBenchmark(
                (landmarks) => {
                    const r = recognizeFn(landmarks);
                    return r || { gesture: 'Unknown', confidence: 0 };
                },
                testSamples,
                3,
            );
            setSummary(result);
            setIsRunning(false);
        }, 100);
    };

    const handleExportCSV = () => {
        if (!summary) return;
        const csv = exportBenchmarkCSV(summary);
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `signbridge-benchmark-${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const toggleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDir(d => d === 'asc' ? 'desc' : 'asc');
        } else {
            setSortField(field);
            setSortDir('desc');
        }
    };

    const sortedResults = useMemo(() => {
        if (!summary) return [];
        const sorted = [...summary.results];
        sorted.sort((a, b) => {
            const aVal = a[sortField];
            const bVal = b[sortField];
            if (typeof aVal === 'string' && typeof bVal === 'string') {
                return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }
            return sortDir === 'asc' ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
        });
        return sorted;
    }, [summary, sortField, sortDir]);

    // Chart data — only show gestures with data
    const chartData = useMemo(() => {
        if (!summary) return [];
        return summary.results
            .filter(r => r.f1 > 0)
            .map(r => ({ gesture: r.gesture, f1: r.f1, precision: r.precision, recall: r.recall }));
    }, [summary]);

    // Confusion matrix — only show labels with data
    const matrixData = useMemo(() => {
        if (!summary) return { labels: [], matrix: [] as number[][] };
        const activeIndices: number[] = [];
        const activeLabels: string[] = [];
        for (let i = 0; i < summary.labels.length; i++) {
            const row = summary.confusionMatrix[i];
            const hasData = row.some(v => v > 0) ||
                summary.confusionMatrix.some(r => r[i] > 0);
            if (hasData) {
                activeIndices.push(i);
                activeLabels.push(summary.labels[i]);
            }
        }
        const matrix = activeIndices.map(i =>
            activeIndices.map(j => summary.confusionMatrix[i][j])
        );
        return { labels: activeLabels, matrix };
    }, [summary]);

    // Color for confusion matrix cells
    const getCellColor = (value: number, maxVal: number) => {
        if (value === 0) return 'bg-gray-50';
        const intensity = Math.min(value / Math.max(maxVal, 1), 1);
        if (intensity > 0.7) return 'bg-green-500 text-white';
        if (intensity > 0.4) return 'bg-green-300';
        if (intensity > 0.1) return 'bg-green-100';
        return 'bg-yellow-50';
    };

    const maxConfusionVal = useMemo(() => {
        if (!matrixData.matrix.length) return 0;
        return Math.max(...matrixData.matrix.flat());
    }, [matrixData]);

    return (
        <div className="space-y-6">
            {/* Summary Card */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <Activity className="w-5 h-5 text-blue-600" />
                            Benchmark Dashboard
                        </div>
                        <div className="flex gap-2">
                            <Button
                                onClick={handleRunBenchmark}
                                disabled={isRunning || !recognizeFn || !testSamples?.length}
                                size="sm"
                            >
                                {isRunning ? 'Running…' : 'Run Benchmark'}
                            </Button>
                            {summary && (
                                <Button onClick={handleExportCSV} variant="outline" size="sm">
                                    <Download className="w-4 h-4 mr-2" />
                                    Export CSV
                                </Button>
                            )}
                        </div>
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    {!summary ? (
                        <div className="text-center py-8 text-gray-500">
                            <p className="text-sm">
                                {(!recognizeFn || !testSamples?.length)
                                    ? 'Collect test samples and train a model first to run benchmarks.'
                                    : 'Click "Run Benchmark" to evaluate model performance.'}
                            </p>
                        </div>
                    ) : (
                        <div className="grid grid-cols-3 gap-4">
                            <div className="text-center p-4 bg-blue-50 rounded-lg">
                                <Target className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-blue-700">
                                    {Math.round(summary.overallAccuracy * 100)}%
                                </div>
                                <div className="text-xs text-gray-600">Overall Accuracy</div>
                            </div>
                            <div className="text-center p-4 bg-green-50 rounded-lg">
                                <TrendingUp className="w-6 h-6 text-green-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-green-700">
                                    {Math.round(summary.meanAveragePrecision * 100)}%
                                </div>
                                <div className="text-xs text-gray-600">Mean Avg Precision</div>
                            </div>
                            <div className="text-center p-4 bg-purple-50 rounded-lg">
                                <Clock className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-purple-700">
                                    {summary.avgLatencyMs.toFixed(1)}ms
                                </div>
                                <div className="text-xs text-gray-600">Avg Latency</div>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* F1 Bar Chart */}
            {summary && chartData.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">F1 Score by Gesture</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey="gesture"
                                    angle={-45}
                                    textAnchor="end"
                                    interval={0}
                                    tick={{ fontSize: 10 }}
                                />
                                <YAxis domain={[0, 1]} />
                                <Tooltip />
                                <Bar dataKey="f1" fill="#3b82f6" name="F1 Score" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            )}

            {/* Sortable Results Table */}
            {summary && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">Per-Gesture Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-x-auto">
                            <table className="w-full text-xs">
                                <thead>
                                    <tr className="border-b">
                                        {([
                                            ['gesture', 'Gesture'],
                                            ['precision', 'Precision'],
                                            ['recall', 'Recall'],
                                            ['f1', 'F1'],
                                            ['avgLatencyMs', 'Latency'],
                                        ] as [SortField, string][]).map(([field, label]) => (
                                            <th
                                                key={field}
                                                className="p-2 text-left cursor-pointer hover:bg-gray-50"
                                                onClick={() => toggleSort(field)}
                                            >
                                                <span className="flex items-center gap-1">
                                                    {label}
                                                    <ArrowUpDown className="w-3 h-3 text-gray-400" />
                                                </span>
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {sortedResults.map(r => (
                                        <tr key={r.gesture} className="border-b hover:bg-gray-50">
                                            <td className="p-2 font-medium">{r.gesture}</td>
                                            <td className="p-2">{(r.precision * 100).toFixed(1)}%</td>
                                            <td className="p-2">{(r.recall * 100).toFixed(1)}%</td>
                                            <td className="p-2">
                                                <Badge
                                                    variant="secondary"
                                                    className={`text-xs ${r.f1 >= 0.8 ? 'bg-green-100 text-green-700'
                                                            : r.f1 >= 0.5 ? 'bg-yellow-100 text-yellow-700'
                                                                : 'bg-red-100 text-red-700'
                                                        }`}
                                                >
                                                    {(r.f1 * 100).toFixed(1)}%
                                                </Badge>
                                            </td>
                                            <td className="p-2">{r.avgLatencyMs.toFixed(1)}ms</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Confusion Matrix */}
            {summary && matrixData.labels.length > 0 && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">Confusion Matrix</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="overflow-x-auto">
                            <div className="text-xs text-gray-500 mb-2">
                                Rows = Actual, Columns = Predicted
                            </div>
                            <div
                                className="grid gap-px bg-gray-200 inline-grid"
                                style={{
                                    gridTemplateColumns: `60px repeat(${matrixData.labels.length}, 36px)`,
                                }}
                            >
                                {/* Header row */}
                                <div className="bg-white p-1 text-xs text-center"></div>
                                {matrixData.labels.map(l => (
                                    <div key={`h-${l}`} className="bg-white p-1 text-xs text-center truncate" title={l}>
                                        {l}
                                    </div>
                                ))}

                                {/* Data rows */}
                                {matrixData.matrix.map((row, i) => (
                                    <React.Fragment key={`r-${i}`}>
                                        <div className="bg-white p-1 text-xs truncate" title={matrixData.labels[i]}>
                                            {matrixData.labels[i]}
                                        </div>
                                        {row.map((val, j) => (
                                            <div
                                                key={`c-${i}-${j}`}
                                                className={`p-1 text-xs text-center ${getCellColor(val, maxConfusionVal)}`}
                                                title={`${matrixData.labels[i]} → ${matrixData.labels[j]}: ${val}`}
                                            >
                                                {val || ''}
                                            </div>
                                        ))}
                                    </React.Fragment>
                                ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
