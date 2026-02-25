import React, { useState, useEffect } from 'react';
import { socket, connectSocket, disconnectSocket } from './socket';
import CameraFeed from './components/CameraFeed';
import ConfidenceChart from './components/ConfidenceChart';
import TextBox from './components/TextBox';

function App() {
    const [isConnected, setIsConnected] = useState(false);
    const [currentPrediction, setCurrentPrediction] = useState('NOTHING');
    const [confidence, setConfidence] = useState(0);
    const [sentence, setSentence] = useState('');
    const [topPredictions, setTopPredictions] = useState([]);
    const [autoSpeak, setAutoSpeak] = useState(false);
    const [stats, setStats] = useState({
        framesProcessed: 0,
        avgConfidence: 0
    });

    useEffect(() => {
        // Connect to socket on mount
        connectSocket();

        // Socket event listeners
        socket.on('connect', () => {
            console.log('Connected to server');
            setIsConnected(true);
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            setIsConnected(false);
        });

        socket.on('connection_status', (data) => {
            console.log('Connection status:', data);
        });

        socket.on('prediction', (data) => {
            if (data.hand_detected) {
                setCurrentPrediction(data.stable_prediction);
                setConfidence(data.stable_confidence);
                setSentence(data.sentence);
                setTopPredictions(data.top_3 || []);

                // Update stats
                setStats(prev => ({
                    framesProcessed: prev.framesProcessed + 1,
                    avgConfidence: ((prev.avgConfidence * prev.framesProcessed) + data.raw_confidence) / (prev.framesProcessed + 1)
                }));
            } else {
                // No hand detected - reset predictions immediately
                setCurrentPrediction('NOTHING');
                setConfidence(0);
                setTopPredictions([]);
                setSentence(data.sentence);
            }
        });

        socket.on('text_cleared', () => {
            setSentence('');
        });

        socket.on('tts_complete', (data) => {
            console.log('TTS complete:', data.text);
        });

        socket.on('tts_error', (data) => {
            console.error('TTS error:', data.message);
            alert(`TTS Error: ${data.message}`);
        });

        socket.on('error', (data) => {
            console.error('Server error:', data.message);
        });

        // Cleanup on unmount
        return () => {
            disconnectSocket();
            socket.off('connect');
            socket.off('disconnect');
            socket.off('connection_status');
            socket.off('prediction');
            socket.off('text_cleared');
            socket.off('tts_complete');
            socket.off('tts_error');
            socket.off('error');
        };
    }, []);

    const handleFrame = (frameData) => {
        if (isConnected) {
            socket.emit('video_frame', { frame: frameData });
        }
    };

    const handleClearText = () => {
        // INSTANT: Clear UI immediately (optimistic update)
        setSentence('');
        // Then notify backend
        socket.emit('clear_text', {});
    };

    const handleSpeakText = (text) => {
        // INSTANT: Send to backend immediately
        socket.emit('speak_text', { text });
    };

    const handleReset = () => {
        socket.emit('reset_session', {});
        setSentence('');
        setCurrentPrediction('NOTHING');
        setConfidence(0);
        setTopPredictions([]);
        setStats({ framesProcessed: 0, avgConfidence: 0 });
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-cyber-darker via-cyber-dark to-cyber-darker">
            {/* Header */}
            <header className="border-b border-cyber-blue/20 bg-cyber-dark/50 backdrop-blur-xl">
                <div className="container mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-gradient-to-br from-cyber-blue to-cyber-purple rounded-lg flex items-center justify-center">
                                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
                                </svg>
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold bg-gradient-to-r from-cyber-blue to-cyber-purple bg-clip-text text-transparent">
                                    Sign Language Translator
                                </h1>
                                <p className="text-sm text-gray-400">Real-time AI-powered gesture recognition</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4">
                            {/* Auto-speak toggle */}
                            <label className="flex items-center gap-2 cursor-pointer">
                                <span className="text-sm text-gray-400">Auto-Speak</span>
                                <div className="relative">
                                    <input
                                        type="checkbox"
                                        checked={autoSpeak}
                                        onChange={(e) => setAutoSpeak(e.target.checked)}
                                        className="sr-only peer"
                                    />
                                    <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-cyber-blue rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyber-blue"></div>
                                </div>
                            </label>

                            {/* Reset button */}
                            <button
                                onClick={handleReset}
                                className="px-4 py-2 bg-cyber-dark border border-red-500/50 text-red-400 rounded-lg hover:bg-red-500/10 hover:border-red-500 transition-all"
                            >
                                Reset
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main content */}
            <main className="container mx-auto px-6 py-8">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left column - Camera feed */}
                    <div className="lg:col-span-2 space-y-6">
                        <div className="glass-panel p-6">
                            <div className="aspect-video bg-black rounded-lg overflow-hidden">
                                <CameraFeed onFrame={handleFrame} isConnected={isConnected} />
                            </div>
                        </div>

                        {/* Current prediction display */}
                        <div className="glass-panel p-6">
                            <div className="flex items-center justify-between">
                                <div>
                                    <p className="text-sm text-gray-400 mb-1">Current Gesture</p>
                                    <p className="text-4xl font-bold font-mono text-cyber-green">
                                        {currentPrediction}
                                    </p>
                                </div>
                                <div className="text-right">
                                    <p className="text-sm text-gray-400 mb-1">Confidence</p>
                                    <div className="flex items-center gap-3">
                                        <div className="w-32 h-3 bg-cyber-darker rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-cyber-blue to-cyber-green transition-all duration-300"
                                                style={{ width: `${confidence * 100}%` }}
                                            />
                                        </div>
                                        <p className="text-2xl font-bold text-cyber-blue">
                                            {(confidence * 100).toFixed(0)}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Text box */}
                        <TextBox
                            text={sentence}
                            onClear={handleClearText}
                            onSpeak={handleSpeakText}
                            autoSpeak={autoSpeak}
                        />
                    </div>

                    {/* Right column - Stats and chart */}
                    <div className="space-y-6">
                        {/* Stats */}
                        <div className="glass-panel p-6">
                            <h2 className="text-xl font-bold text-cyber-blue mb-4">Statistics</h2>
                            <div className="space-y-4">
                                <div>
                                    <p className="text-sm text-gray-400">Frames Processed</p>
                                    <p className="text-2xl font-bold text-white">{stats.framesProcessed}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-gray-400">Avg Confidence</p>
                                    <p className="text-2xl font-bold text-cyber-green">
                                        {(stats.avgConfidence * 100).toFixed(1)}%
                                    </p>
                                </div>
                                <div>
                                    <p className="text-sm text-gray-400">Words Detected</p>
                                    <p className="text-2xl font-bold text-cyber-purple">
                                        {sentence.split(' ').filter(w => w.length > 0).length}
                                    </p>
                                </div>
                            </div>
                        </div>

                        {/* Confidence chart */}
                        <ConfidenceChart predictions={topPredictions} />
                    </div>
                </div>
            </main>

            {/* Footer */}
            <footer className="border-t border-cyber-blue/20 bg-cyber-dark/50 backdrop-blur-xl mt-12">
                <div className="container mx-auto px-6 py-4">
                    <p className="text-center text-sm text-gray-400">
                        Final Year B.Sc. CS Project • Real-Time Sign Language Translation
                    </p>
                </div>
            </footer>
        </div>
    );
}

export default App;
