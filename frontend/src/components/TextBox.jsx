import React, { useState, useEffect } from 'react';

const TextBox = ({ text, onClear, onSpeak, autoSpeak = false }) => {
    const [lastText, setLastText] = useState('');

    // Auto-speak when text changes (if enabled)
    useEffect(() => {
        if (autoSpeak && text && text !== lastText && text.trim().length > 0) {
            handleSpeak();
            setLastText(text);
        }
    }, [text, autoSpeak]);

    const handleSpeak = () => {
        if (text && text.trim().length > 0) {
            onSpeak(text);
        }
    };

    // Listen for TTS completion from parent
    useEffect(() => {
        // Reset speaking state when text changes (TTS completed)
        const timer = setTimeout(() => setIsSpeaking(false), 100);
        return () => clearTimeout(timer);
    }, [text]);

    const handleClear = () => {
        onClear();
        setLastText('');
    };

    return (
        <div className="glass-panel p-6">
            <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-cyber-purple">
                    Translated Text
                </h2>
            </div>

            {/* Text display area */}
            <div className="relative">
                <div className="min-h-[120px] max-h-[200px] overflow-y-auto p-4 bg-cyber-darker/50 rounded-lg border border-cyber-purple/30 mb-4">
                    {text ? (
                        <p className="text-2xl font-mono text-white leading-relaxed break-words">
                            {text}
                            <span className="inline-block w-1 h-8 ml-1 bg-cyber-purple animate-pulse" />
                        </p>
                    ) : (
                        <p className="text-xl text-gray-500 italic">
                            Start signing to see text appear here...
                        </p>
                    )}
                </div>

                {/* Character count */}
                <div className="absolute bottom-6 right-6 text-xs text-gray-500">
                    {text.length} characters
                </div>
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
                <button
                    onClick={handleSpeak}
                    disabled={!text || text.trim().length === 0}
                    className="flex-1 btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                    </svg>
                    Speak Text
                </button>

                <button
                    onClick={handleClear}
                    disabled={!text || text.trim().length === 0}
                    className="flex-1 btn-secondary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Clear Text
                </button>
            </div>
        </div>
    );
};

export default TextBox;
