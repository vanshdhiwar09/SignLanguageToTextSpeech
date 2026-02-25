import React, { useRef, useEffect, useState } from 'react';

const CameraFeed = ({ onFrame, isConnected }) => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [cameraActive, setCameraActive] = useState(false);
    const [error, setError] = useState(null);
    const streamRef = useRef(null);

    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, []);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                streamRef.current = stream;
                setCameraActive(true);
                setError(null);
            }
        } catch (err) {
            console.error('Camera error:', err);
            setError('Failed to access camera. Please grant camera permissions.');
            setCameraActive(false);
        }
    };

    const stopCamera = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        setCameraActive(false);
    };

    useEffect(() => {
        if (!cameraActive || !isConnected) return;

        const interval = setInterval(() => {
            captureAndSendFrame();
        }, 100); // Send frame every 100ms (10 FPS)

        return () => clearInterval(interval);
    }, [cameraActive, isConnected]);

    const captureAndSendFrame = () => {
        if (!videoRef.current || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const video = videoRef.current;
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Mirror the canvas to match the flipped video display
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();

        // Convert to base64
        const frameData = canvas.toDataURL('image/jpeg', 0.8);

        // Send to parent
        if (onFrame) {
            onFrame(frameData);
        }
    };

    return (
        <div className="relative w-full h-full">
            {/* Video element (hidden, used for capture) */}
            <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover rounded-lg"
                style={{ transform: 'scaleX(-1)' }}
            />

            {/* Canvas for drawing (overlay) */}
            <canvas
                ref={canvasRef}
                className="hidden"
            />

            {/* Status indicators */}
            <div className="absolute top-4 left-4 flex items-center gap-3">
                <div className={`flex items-center gap-2 px-4 py-2 rounded-lg glass-panel ${cameraActive ? 'border-cyber-green' : 'border-red-500'
                    } border`}>
                    <div className={`w-3 h-3 rounded-full ${cameraActive ? 'bg-cyber-green animate-pulse' : 'bg-red-500'
                        }`} />
                    <span className="text-sm font-medium">
                        {cameraActive ? 'Camera Active' : 'Camera Inactive'}
                    </span>
                </div>

                <div className={`flex items-center gap-2 px-4 py-2 rounded-lg glass-panel ${isConnected ? 'border-cyber-blue' : 'border-yellow-500'
                    } border`}>
                    <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-cyber-blue animate-pulse' : 'bg-yellow-500'
                        }`} />
                    <span className="text-sm font-medium">
                        {isConnected ? 'Connected' : 'Disconnected'}
                    </span>
                </div>
            </div>

            {/* Error message */}
            {error && (
                <div className="absolute bottom-4 left-4 right-4 bg-red-500/20 border border-red-500 rounded-lg p-4">
                    <p className="text-red-200 text-sm">{error}</p>
                    <button
                        onClick={startCamera}
                        className="mt-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                    >
                        Retry
                    </button>
                </div>
            )}
        </div>
    );
};

export default CameraFeed;
