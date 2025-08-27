import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
import { Camera, useCameraDevice } from 'react-native-vision-camera';
import TextRecognition from '@react-native-ml-kit/text-recognition';

export const ReactNativeMLKitPage = () => {
  const [hasPermission, setHasPermission] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  const device = useCameraDevice('back');
  const cameraRef = useRef<Camera>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'granted');
    })();
  }, []);
/**
 * 對指定圖片路徑執行 OCR，回傳所有文字
 */
 async function runOCR(photo: { path: string }): Promise<string> {
  try {
    const result = await TextRecognition.recognize(`file://${photo.path}`);
    return result.blocks.map((block) => block.text).join('\n');
  } catch (err) {
    console.error('OCR failed:', err);
    throw err;
  }
}
  const takePhotoAndRecognize = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);

      // 拍照
      const photo = await cameraRef.current.takePhoto();

      // OCR 辨識
      const allText = await runOCR(photo);

      setRecognizedText(allText || '(沒有辨識到文字)');
    } catch (err) {
      console.error('OCR failed:', err);
      setRecognizedText('辨識失敗：' + String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  if (!device || !hasPermission) {
    return <Text style={styles.loadingText}>相機初始化中...</Text>;
  }

  return (
    <View style={styles.container}>
      <Camera ref={cameraRef} style={StyleSheet.absoluteFill} device={device} isActive={true} photo={true} />
      <View style={styles.overlay}>
        <TouchableOpacity
          style={[styles.button, isProcessing && styles.buttonDisabled]}
          onPress={takePhotoAndRecognize}
          disabled={isProcessing}
        >
          <Text style={styles.buttonText}>{isProcessing ? '辨識中...' : '拍照並辨識'}</Text>
        </TouchableOpacity>
        <View style={styles.resultBox}>
          <Text style={styles.ocrText}>辨識結果：</Text>
          <Text style={styles.ocrResult}>{recognizedText}</Text>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  loadingText: {
    flex: 1,
    textAlign: 'center',
    textAlignVertical: 'center',
    fontSize: 16,
  },
  overlay: {
    position: 'absolute',
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.6)',
    padding: 12,
    width: '100%',
  },
  button: {
    backgroundColor: '#2196F3',
    paddingVertical: 10,
    borderRadius: 6,
    marginBottom: 10,
  },
  buttonDisabled: {
    backgroundColor: '#888',
  },
  buttonText: {
    color: '#fff',
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 16,
  },
  resultBox: {
    maxHeight: 200,
  },
  ocrText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  ocrResult: {
    color: '#fff',
    marginTop: 6,
  },
});
