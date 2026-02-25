import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const ConfidenceChart = ({ predictions = [] }) => {
    // Format data for chart
    const chartData = predictions.map(pred => ({
        label: pred.label,
        confidence: (pred.confidence * 100).toFixed(1)
    }));

    // Color gradient based on confidence
    const getColor = (confidence) => {
        if (confidence >= 85) return '#00ff88'; // cyber-green
        if (confidence >= 70) return '#00d9ff'; // cyber-blue
        if (confidence >= 50) return '#b537f2'; // cyber-purple
        return '#ff2e97'; // cyber-pink
    };

    return (
        <div className="glass-panel p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-cyber-blue">
                    Top Predictions
                </h2>
                <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-cyber-green animate-pulse" />
                    <span className="text-sm text-gray-400">Live</span>
                </div>
            </div>

            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                        data={chartData}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1a1f3a" />
                        <XAxis
                            type="number"
                            domain={[0, 100]}
                            stroke="#6b7280"
                            tick={{ fill: '#9ca3af' }}
                        />
                        <YAxis
                            type="category"
                            dataKey="label"
                            stroke="#6b7280"
                            tick={{ fill: '#9ca3af', fontSize: 14, fontWeight: 600 }}
                            width={100}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: '#0a0e27',
                                border: '1px solid #00d9ff',
                                borderRadius: '8px',
                                color: '#fff'
                            }}
                            formatter={(value) => [`${value}%`, 'Confidence']}
                        />
                        <Bar dataKey="confidence" radius={[0, 8, 8, 0]}>
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={getColor(parseFloat(entry.confidence))} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            ) : (
                <div className="h-64 flex items-center justify-center">
                    <div className="text-center">
                        <div className="w-16 h-16 mx-auto mb-4 border-4 border-cyber-blue border-t-transparent rounded-full animate-spin" />
                        <p className="text-gray-400">Waiting for predictions...</p>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-sm bg-cyber-green" />
                    <span className="text-gray-400">≥85% Excellent</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-sm bg-cyber-blue" />
                    <span className="text-gray-400">≥70% Good</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-sm bg-cyber-purple" />
                    <span className="text-gray-400">≥50% Fair</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-sm bg-cyber-pink" />
                    <span className="text-gray-400">&lt;50% Low</span>
                </div>
            </div>
        </div>
    );
};

export default ConfidenceChart;
