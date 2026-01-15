import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from './ui/badge';
import { Progress } from '../components/ui/progress';
import { Camera, CameraOff, RotateCcw, Play, Pause, AlertCircle, Info, Volume2, VolumeX } from 'lucide-react';

interface RecognizedSign {
  sign: string;
  confidence: number;
  timestamp: Date;
}

// Sign language gesture patterns based on hand landmarks
// These are simplified patterns - in production, you'd use a trained ML model
const signPatterns = {
  'Hello': (landmarks: number[][]) => {
    const fingersExtended = checkFingersExtended(landmarks);
    return fingersExtended >= 4 ? 0.85 : 0;
  },
  'Thank you': (landmarks: number[][]) => {
    const palmFacingUp = checkPalmOrientation(landmarks, 'up');
    const fingersExtended = checkFingersExtended(landmarks);
    return (palmFacingUp && fingersExtended >= 3) ? 0.80 : 0;
  },
  'Please': (landmarks: number[][]) => {
    const palmOpen = checkFingersExtended(landmarks) >= 4;
    return palmOpen ? 0.75 : 0;
  },
  'Yes': (landmarks: number[][]) => {
    const fistClosed = checkFingersExtended(landmarks) <= 1;
    return fistClosed ? 0.82 : 0;
  },
  'No': (landmarks: number[][]) => {
    const twoFingersExtended = checkFingersExtended(landmarks) === 2;
    return twoFingersExtended ? 0.78 : 0;
  },
  'Good': (landmarks: number[][]) => {
    const thumbUp = checkThumbPosition(landmarks, 'up');
    const fingersDown = checkFingersExtended(landmarks) <= 1;
    return (thumbUp && fingersDown) ? 0.88 : 0;
  },
  'Bad': (landmarks: number[][]) => {
    const thumbDown = checkThumbPosition(landmarks, 'down');
    const fingersDown = checkFingersExtended(landmarks) <= 1;
    return (thumbDown && fingersDown) ? 0.83 : 0;
  },
  'Help': (landmarks: number[][]) => {
    const fingersExtended = checkFingersExtended(landmarks);
    return fingersExtended >= 3 ? 0.76 : 0;
  },
  'Sorry': (landmarks: number[][]) => {
    const fistClosed = checkFingersExtended(landmarks) <= 1;
    return fistClosed ? 0.79 : 0;
  },
  'Love': (landmarks: number[][]) => {
    const specialPattern = checkLoveSign(landmarks);
    return specialPattern ? 0.90 : 0;
  }
};

// Helper functions for gesture recognition
function checkFingersExtended(landmarks: number[][]): number {
  if (!landmarks || landmarks.length < 21) return 0;
  
  let extendedCount = 0;
  const fingerTips = [8, 12, 16, 20];
  const fingerMCPs = [5, 9, 13, 17];
  
  fingerTips.forEach((tip, i) => {
    const tipY = landmarks[tip][1];
    const mcpY = landmarks[fingerMCPs[i]][1];
    if (tipY < mcpY - 20) {
      extendedCount++;
    }
  });
  
  const thumbTip = landmarks[4];
  const thumbMCP = landmarks[2];
  if (Math.abs(thumbTip[0] - thumbMCP[0]) > 30) {
    extendedCount++;
  }
  
  return extendedCount;
}

function checkPalmOrientation(landmarks: number[][], direction: string): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const wrist = landmarks[0];
  const middleMCP = landmarks[9];
  if (direction === 'up') {
    return middleMCP[1] < wrist[1] - 30;
  }
  return false;
}

function checkThumbPosition(landmarks: number[][], direction: string): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const thumbTip = landmarks[4];
  const indexMCP = landmarks[5];
  if (direction === 'up') {
    return thumbTip[1] < indexMCP[1] - 40;
  } else if (direction === 'down') {
    return thumbTip[1] > indexMCP[1] + 40;
  }
  return false;
}

function checkLoveSign(landmarks: number[][]): boolean {
  if (!landmarks || landmarks.length < 21) return false;
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinkyTip = landmarks[20];
  const middleTip = landmarks[12];
  const thumbMCP = landmarks[2];
  const indexMCP = landmarks[5];
  const middleMCP = landmarks[9];
  
  const thumbExtended = Math.abs(thumbTip[0] - thumbMCP[0]) > 30;
  const indexExtended = indexTip[1] < indexMCP[1] - 20;
  const middleDown = middleTip[1] > middleMCP[1];
  const pinkyExtended = pinkyTip[1] < middleMCP[1] - 10;
  
  return thumbExtended && indexExtended && pinkyExtended && middleDown;
}

function recognizeSign(landmarks: number[][]): RecognizedSign | null {
  let bestMatch: RecognizedSign | null = null;
  let highestConfidence = 0.70;
  
  for (const [signName, detector] of Object.entries(signPatterns)) {
    const confidence = detector(landmarks);
    if (confidence > highestConfidence) {
      highestConfidence = confidence;
      bestMatch = {
        sign: signName,
        confidence: confidence,
        timestamp: new Date()
      };
    }
  }
  return bestMatch;
}

export function SignLanguageRecognizer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [currentSign, setCurrentSign] = useState<RecognizedSign | null>(null);
  const [recognitionHistory, setRecognitionHistory] = useState<RecognizedSign[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [audioEnabled, setAudioEnabled] = useState(true);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const recognitionIntervalRef = useRef<number | null>(null);

  // Ensure video element always receives the stream
  useEffect(() => {
    if (cameraActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play().catch(e => console.error('Play error:', e));
        setVideoError(null);
      };
      videoRef.current.onerror = () => {
        setVideoError('Video playback failed.');
      };
      videoRef.current.onplay = () => {
        setVideoError(null);
      };
    }
  }, [stream, cameraActive]);

  // Load the handpose model
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelLoading(true);
        // Dynamically import TensorFlow.js and handpose
        const tf = await import('@tensorflow/tfjs');
        const handpose = await import('@tensorflow-models/handpose');
        await tf.ready();
        const loadedModel = await handpose.load();
        setModel(loadedModel);
        setModelLoading(false);
      } catch (error) {
        setCameraError('Error loading AI model. Please refresh and try again.');
        setModelLoading(false);
        console.error('Error loading model:', error);
      }
    };
    loadModel();
  }, []);

  const speakText = useCallback((text: string) => {
    if (!audioEnabled) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
    }
  }, [audioEnabled]);

  const replayAudio = () => {
    if (currentSign) {
      speakText(currentSign.sign);
    }
  };

  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setCameraError(null);
      
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported by this browser');
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user'
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        setStream(mediaStream);
        setCameraActive(true);
        
        // Ensure video playback starts
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(e => console.error('Play error:', e));
          setVideoError(null);
        };
        videoRef.current.onerror = () => {
          setVideoError('Video playback failed.');
        };
        videoRef.current.onplay = () => {
          setVideoError(null);
        };
      } else {
        // If videoRef is null, just set the stream state and let the useEffect handle it
        setStream(mediaStream);
        setCameraActive(true);
      }
    } catch (error: any) {
      if (error.name !== 'NotAllowedError') {
        console.error('Error accessing camera:', error);
      }
      
      let errorMessage = 'Could not access camera. ';
      
      if (error.name === 'NotAllowedError') {
        errorMessage += 'Camera permission was denied. Please allow camera access in your browser settings and try again.';
        setTimeout(() => {
          setDemoMode(true);
          setCameraError(null);
          setIsRecognizing(true);
        }, 500);
      } else if (error.name === 'NotFoundError') {
        errorMessage += 'No camera found on this device.';
      } else if (error.name === 'NotReadableError') {
        errorMessage += 'Camera is already in use by another application.';
      } else if (error.name === 'OverconstrainedError') {
        errorMessage += 'Camera does not meet the required specifications.';
      } else if (error.message === 'Camera not supported by this browser') {
        errorMessage += 'Your browser does not support camera access.';
      } else {
        errorMessage += 'Please check your camera and permissions.';
      }
      
      setCameraError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setCameraActive(false);
      setIsRecognizing(false);
    }
    setCameraError(null);
    setDemoMode(false);
  }, [stream]);

  const startDemoMode = useCallback(() => {
    setDemoMode(true);
    setCameraError(null);
    setIsRecognizing(true);
    
    const interval = setInterval(() => {
      const mockSigns = ['Hello', 'Thank you', 'Please', 'Yes', 'No', 'Good', 'Bad', 'Help', 'Sorry', 'Love'];
      const randomSign = mockSigns[Math.floor(Math.random() * mockSigns.length)];
      const recognizedSign: RecognizedSign = {
        sign: randomSign,
        confidence: 0.75 + Math.random() * 0.2,
        timestamp: new Date()
      };
      
      setCurrentSign(recognizedSign);
      setRecognitionHistory(prev => [recognizedSign, ...prev].slice(0, 10));
    }, 2000);

    recognitionIntervalRef.current = interval as unknown as number;
  }, []);

  const startRecognition = useCallback(async () => {
    if (!model && !demoMode) {
      setCameraError('Model is still loading. Please wait...');
      return;
    }

    setIsRecognizing(true);
    
    if (demoMode) {
      const interval = setInterval(() => {
        const mockSigns = ['Hello', 'Thank you', 'Please', 'Yes', 'No', 'Good', 'Bad', 'Help', 'Sorry', 'Love'];
        const randomSign = mockSigns[Math.floor(Math.random() * mockSigns.length)];
        const recognizedSign: RecognizedSign = {
          sign: randomSign,
          confidence: 0.75 + Math.random() * 0.2,
          timestamp: new Date()
        };
        
        setCurrentSign(recognizedSign);
        setRecognitionHistory(prev => [recognizedSign, ...prev].slice(0, 10));
      }, 2000);

      recognitionIntervalRef.current = interval as unknown as number;
      return;
    }

    let isActive = true;

    const detectHands = async () => {
      if (videoRef.current && model && isActive) {
        try {
          const predictions = await model.estimateHands(videoRef.current);
          if (predictions.length > 0) {
            const landmarks = predictions[0].landmarks;
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx && videoRef.current) {
                canvasRef.current.width = videoRef.current.videoWidth;
                canvasRef.current.height = videoRef.current.videoHeight;
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                ctx.fillStyle = 'red';
                landmarks.forEach((landmark: number[]) => {
                  ctx.beginPath();
                  ctx.arc(landmark[0], landmark[1], 5, 0, 2 * Math.PI);
                  ctx.fill();
                });
                ctx.strokeStyle = 'blue';
                ctx.lineWidth = 2;
                const connections = [
                  [0,1],[1,2],[2,3],[3,4],
                  [0,5],[5,6],[6,7],[7,8],
                  [0,9],[9,10],[10,11],[11,12],
                  [0,13],[13,14],[14,15],[15,16],
                  [0,17],[17,18],[18,19],[19,20],
                  [5,9],[9,13],[13,17]
                ];
                connections.forEach(([start, end]) => {
                  ctx.beginPath();
                  ctx.moveTo(landmarks[start][0], landmarks[start][1]);
                  ctx.lineTo(landmarks[end][0], landmarks[end][1]);
                  ctx.stroke();
                });
              }
            }
            const recognizedSign = recognizeSign(landmarks);
            if (recognizedSign) {
              setCurrentSign(recognizedSign);
              setRecognitionHistory(prev => {
                if (prev.length > 0 && prev[0].sign === recognizedSign.sign) {
                  return prev;
                }
                return [recognizedSign, ...prev].slice(0, 10);
              });
            }
          } else {
            if (canvasRef.current) {
              const ctx = canvasRef.current.getContext('2d');
              if (ctx) {
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
              }
            }
            setCurrentSign(null);
          }
        } catch (error) {
          setCameraError('Error detecting hands: ' + (error instanceof Error ? error.message : String(error)));
          console.error('Error detecting hands:', error);
        }
      }
      
      if (isActive) {
        requestAnimationFrame(detectHands);
      }
    };

    detectHands();

    return () => {
      isActive = false;
    };
  }, [model, demoMode]);

  const stopRecognition = useCallback(() => {
    setIsRecognizing(false);
    if (recognitionIntervalRef.current) {
      clearInterval(recognitionIntervalRef.current);
      recognitionIntervalRef.current = null;
    }
    
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
    
    setCurrentSign(null);
  }, []);

  const clearHistory = () => {
    setRecognitionHistory([]);
  };

  useEffect(() => {
    return () => {
      stopCamera();
      if (recognitionIntervalRef.current) {
        clearInterval(recognitionIntervalRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (currentSign && audioEnabled) {
      speakText(currentSign.sign);
    }
  }, [currentSign, audioEnabled, speakText]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Sign Language Recognizer
          </h1>
          <p className="text-gray-600 dark:text-gray-300">
            Real-time ASL detection using AI-powered hand tracking
          </p>
        </div>

        {modelLoading && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg dark:bg-blue-950/20 dark:border-blue-800/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5 animate-pulse" />
              <div>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Loading AI model for hand detection... This may take a moment.
                </p>
              </div>
            </div>
          </div>
        )}

        {!cameraActive && !demoMode && !cameraError && (
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg dark:bg-blue-950/20 dark:border-blue-800/30">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  New to sign language recognition?
                </p>
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  Try our Demo Mode to see how recognition works without needing camera access. 
                  Perfect for exploring the app or if you're having camera issues.
                </p>
                <Button 
                  onClick={startDemoMode} 
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <Play className="w-4 h-4 mr-2" />
                  Start Demo Mode
                </Button>
              </div>
            </div>
          </div>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Camera Feed {demoMode && <Badge variant="secondary">Demo Mode</Badge>}
              <div className="flex gap-2">
                {!cameraActive && !demoMode ? (
                  <>
                    <Button 
                      onClick={startCamera} 
                      disabled={isLoading}
                      variant="outline"
                      size="sm"
                    >
                      <Camera className="w-4 h-4 mr-2" />
                      {isLoading ? 'Starting...' : 'Start Camera'}
                    </Button>
                    <Button 
                      onClick={startDemoMode}
                      variant="secondary"
                      size="sm"
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Demo Mode
                    </Button>
                  </>
                ) : cameraActive ? (
                  <Button 
                    onClick={stopCamera}
                    variant="outline"
                    size="sm"
                  >
                    <CameraOff className="w-4 h-4 mr-2" />
                    Stop Camera
                  </Button>
                ) : (
                  <Button 
                    onClick={() => setDemoMode(false)}
                    variant="outline"
                    size="sm"
                  >
                    Exit Demo
                  </Button>
                )}
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative bg-black rounded-lg overflow-hidden">
              {cameraActive ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-80 object-cover z-10 relative"
                    style={{ 
                      background: 'black', 
                      display: 'block',
                      width: '100%',
                      height: '320px',
                      objectFit: 'cover'
                    }}
                  >
                    Sorry, your browser does not support embedded videos.
                  </video>
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full z-20"
                    style={{ 
                      pointerEvents: 'none',
                      position: 'absolute',
                      top: 0,
                      left: 0
                    }}
                  />
                  {/* Fallback if video cannot play */}
                  {videoError && (
                    <div className="absolute inset-0 flex items-center justify-center text-white bg-black bg-opacity-80 z-30">
                      <p>{videoError}</p>
                    </div>
                  )}
                </>
              ) : demoMode ? (
                <div className="w-full h-80 flex items-center justify-center bg-gradient-to-br from-blue-900 to-purple-900 text-white">
                  <div className="text-center">
                    <Play className="w-16 h-16 mx-auto mb-4" />
                    <p className="text-lg mb-2">Demo Mode Active</p>
                    <p className="text-sm opacity-80">Simulating sign recognition without camera</p>
                  </div>
                </div>
              ) : (
                <div className="w-full h-80 flex items-center justify-center text-muted-foreground">
                  <div className="text-center space-y-4">
                    <div>
                      <Camera className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Click "Start Camera" to begin</p>
                    </div>
                    {!cameraError && (
                      <div className="text-center">
                        <p className="text-xs opacity-60 mb-3">No camera? No problem!</p>
                        <Button 
                          onClick={startDemoMode} 
                          variant="secondary" 
                          size="sm"
                        >
                          <Play className="w-4 h-4 mr-2" />
                          Try Demo Mode
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {isRecognizing && (
                <div className="absolute top-4 left-4">
                  <Badge variant="destructive" className="animate-pulse">
                    ● {demoMode ? 'DEMO MODE' : 'RECOGNIZING'}
                  </Badge>
                </div>
              )}
            </div>
            
            {cameraError && (
              <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm text-destructive">Camera/Recognition Error</p>
                      <p className="text-sm text-destructive/80">{cameraError}</p>
                    </div>
                    <div className="text-xs text-destructive/60 space-y-1">
                      <p><strong>To enable camera:</strong></p>
                      <ul className="list-disc list-inside space-y-0.5 ml-2">
                        <li>Click the camera icon in your browser's address bar</li>
                        <li>Select "Allow" when prompted for camera permission</li>
                        <li>Refresh the page if needed</li>
                      </ul>
                    </div>
                    <div className="pt-2 border-t border-destructive/20">
                      <p className="text-xs text-destructive/80 mb-2">
                        <strong>Can't enable camera? Try our demo mode instead:</strong>
                      </p>
                      <Button 
                        onClick={startDemoMode} 
                        variant="outline" 
                        size="sm"
                        className="border-destructive/30 text-destructive hover:bg-destructive/10"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Start Demo Mode
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="flex gap-2 mt-4 flex-wrap">
              {cameraActive ? (
                <>
                  {!isRecognizing ? (
                    <Button onClick={startRecognition}>
                      <Play className="w-4 h-4 mr-2" />
                      Start Recognition
                    </Button>
                  ) : (
                    <Button onClick={stopRecognition} variant="destructive">
                      <Pause className="w-4 h-4 mr-2" />
                      Stop Recognition
                    </Button>
                  )}
                </>
              ) : demoMode ? (
                <>
                  {!isRecognizing ? (
                    <Button onClick={startRecognition}>
                      <Play className="w-4 h-4 mr-2" />
                      Start Demo Recognition
                    </Button>
                  ) : (
                    <Button onClick={stopRecognition} variant="destructive">
                      <Pause className="w-4 h-4 mr-2" />
                      Stop Demo
                    </Button>
                  )}
                  <Button onClick={() => setDemoMode(false)} variant="outline">
                    Exit Demo
                  </Button>
                  <Button 
                    onClick={startCamera} 
                    variant="outline"
                    size="sm"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    Try Camera Again
                  </Button>
                </>
              ) : (
                <div className="flex gap-2 flex-wrap">
                  {!cameraError && (
                    <Button onClick={startDemoMode} variant="secondary">
                      <Play className="w-4 h-4 mr-2" />
                      Demo Mode
                    </Button>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {currentSign && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                Current Recognition
                {demoMode && (
                  <Badge variant="secondary" className="text-xs">
                    Demo
                  </Badge>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">{currentSign.sign}</div>
                  <div className="text-sm text-muted-foreground mb-4">
                    {demoMode ? 'Demo detected' : 'Detected'} at {currentSign.timestamp.toLocaleTimeString()}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Confidence</span>
                    <span>{Math.round(currentSign.confidence * 100)}%</span>
                  </div>
                  <Progress value={currentSign.confidence * 100} />
                </div>
                
                {demoMode && (
                  <div className="text-xs text-muted-foreground text-center mt-3 p-2 bg-muted/50 rounded">
                    This is a simulation. Enable camera access for real recognition.
                  </div>
                )}
                
                <div className="flex items-center justify-center gap-2 mt-4">
                  <Button 
                    onClick={replayAudio} 
                    variant="outline" 
                    size="sm"
                    disabled={!audioEnabled || isSpeaking}
                  >
                    {isSpeaking ? (
                      <>
                        <VolumeX className="w-4 h-4 mr-2" />
                        Speaking...
                      </>
                    ) : (
                      <>
                        <Volume2 className="w-4 h-4 mr-2" />
                        Replay Audio
                      </>
                    )}
                  </Button>
                  <Button 
                    onClick={() => setAudioEnabled(!audioEnabled)} 
                    variant={audioEnabled ? "outline" : "secondary"}
                    size="sm"
                  >
                    {audioEnabled ? (
                      <>
                        <Volume2 className="w-4 h-4 mr-2" />
                        Audio On
                      </>
                    ) : (
                      <>
                        <VolumeX className="w-4 h-4 mr-2" />
                        Audio Off
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Recognition History
              <Button 
                onClick={clearHistory} 
                variant="outline" 
                size="sm"
                disabled={recognitionHistory.length === 0}
              >
                <RotateCcw className="w-4 h-4 mr-2" />
                Clear
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {recognitionHistory.length === 0 ? (
              <p className="text-muted-foreground text-center py-8">
                No signs recognized yet. {demoMode ? 'Start demo recognition' : 'Start the camera and recognition'} to begin.
              </p>
            ) : (
              <div className="space-y-3">
                {recognitionHistory.map((item, index) => (
                  <div 
                    key={`${item.sign}-${item.timestamp.getTime()}`}
                    className="flex items-center justify-between p-3 bg-muted rounded-lg"
                  >
                    <div>
                      <div>{item.sign}</div>
                      <div className="text-sm text-muted-foreground">
                        {item.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                    <Badge variant="secondary">
                      {Math.round(item.confidence * 100)}%
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Sign Language Guide</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <p className="text-muted-foreground mb-4">
                Position your hand clearly in front of the camera. Here are tips for each sign:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Hello:</strong> Open hand, all fingers extended (wave)
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Thank You:</strong> Open hand, palm up, fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Please:</strong> Open palm with fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Yes:</strong> Closed fist (nodding motion)
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>No:</strong> Index and middle fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Good:</strong> Thumbs up, other fingers closed
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Bad:</strong> Thumbs down, other fingers closed
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Help:</strong> Open hand with fingers extended
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Sorry:</strong> Closed fist
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <strong>Love:</strong> Thumb, index, and pinky extended (ILY sign)
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-4">
                Note: These are simplified gesture patterns. For best results, hold each sign clearly for 1-2 seconds.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}