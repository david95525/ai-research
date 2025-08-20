import { useEffect, useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { loadLabels, nonMaxSuppression, dumpTensor } from '../services/aiService';

export const FastTFLitePage = () => {
  const device = useCameraDevice('back');
  const [hasPermission, setHasPermission] = useState(false);
  const [results, setResults] = useState<{ label: string; prob: number; box?: number[] }[]>([]);
  const [scanStatus, setScanStatus] = useState('正在掃描中...');

  const labels = useSharedValue<string[]>([]);
  const modelRef = useSharedValue<any>(null);
  const lastProcessed = useSharedValue(0);
  const scanning = useSharedValue(true);
  const { resize } = useResizePlugin();

  useEffect(() => {
    (async () => {
      const lbls = await loadLabels();
      labels.value = lbls;
    })();
  }, []);

  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === 'granted');

      try {
        const loadedModel = await loadTensorflowModel({ url: 'model' });
        modelRef.value = loadedModel;
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);
  const Rescan = () => {
    onScanStatus('正在掃描中...');
    setResults([]);
    scanning.value = true;
  };
  const onInferenceResult = Worklets.createRunOnJS((res: typeof results) => {
    setResults(res);
  });
  const onScanStatus = Worklets.createRunOnJS((status: string) => {
    setScanStatus(status);
  });
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      if (!scanning.value) return;
      const model = modelRef.value;
      if (!model || labels.value.length === 0) return;
      const now = Date.now();
      if (now - lastProcessed.value < 1000) return;
      lastProcessed.value = now;

      const input = resize(frame, {
        scale: { width: 192, height: 192 },
        pixelFormat: 'rgb',
        dataType: 'float32',
      });

      try {
        const result = model.runSync([input]);
        // 假設 TFLite 模型輸出 [boxes, scores, classes]
        const boxes = Array.isArray(result[0]) ? (result[0] as number[][]) : [];
        console.log(result.length);
        const scores = result[1] instanceof Float32Array ? (result[1] as Float32Array) : new Float32Array();
        const classes = result[2] instanceof Float32Array ? (result[2] as Float32Array) : new Float32Array();
        if (boxes.length === 0 || scores.length === 0 || classes.length === 0) {
          onScanStatus('模型輸出為空,已停止');
          return;
        }
        const selectedIdxs = nonMaxSuppression(boxes, Array.from(scores));
        const detections = selectedIdxs.map((i) => ({
          label: labels.value[classes[i]] ?? 'unknown',
          prob: scores[i],
          box: boxes[i],
        }));
        onInferenceResult(detections);
        onScanStatus('掃瞄有結果,已停止');
        scanning.value = false;
      } catch (e) {
        console.error('推論錯誤', e);
      }
    },
    [labels, modelRef, lastProcessed],
  );

  if (!device || !hasPermission) return <Text>相機初始化中...</Text>;

  return (
    <View style={styles.container}>
      <Camera style={StyleSheet.absoluteFill} device={device} isActive frameProcessor={frameProcessor} />
      <View style={styles.overlay}>
        <Text style={styles.status}>{scanStatus}</Text>
        {results.map((r, idx) => (
          <Text key={idx} style={styles.result}>
            {r.label} ({(r.prob * 100).toFixed(1)}%)
          </Text>
        ))}
      </View>
      <Button title="重新啟動掃描" onPress={Rescan} />
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  overlay: {
    position: 'absolute',
    bottom: 0,
    width: '100%',
    padding: 16,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  status: { color: 'yellow', fontSize: 16, marginBottom: 8 },
  result: { color: '#fff', fontSize: 18 },
});
