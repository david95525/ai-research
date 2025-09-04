import { useEffect, useState } from 'react';
import { Button, StyleSheet, Text, View, ScrollView } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { loadLabels, predictTflite } from '../utils/index';

export const TFLiteScanPage = () => {
  const device = useCameraDevice('back');
  const [hasPermission, setHasPermission] = useState(false);
  const [results, setResults] = useState<{ label: string; prob: number; box?: number[] }[]>([]);
  const [scanStatus, setScanStatus] = useState('');
  const [trigger, setTrigger] = useState(true);
  const labels = useSharedValue<string[]>([]);
  const modelRef = useSharedValue<TensorflowModel | null>(null);
  const scanTrigger = useSharedValue(0);

  // 加載標籤
  useEffect(() => {
    (async () => {
      const lbls = await loadLabels();
      labels.value = lbls;
    })();
  }, []);

  // 相機權限與模型
  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === 'granted');

      try {
        const loadedModel = await loadTensorflowModel({ url: 'tflitemodel' });
        modelRef.value = loadedModel;
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);

  const onInferenceResult = Worklets.createRunOnJS((res: typeof results) => {
    setResults(res);
  });

  const onScanStatus = Worklets.createRunOnJS((status: string) => {
    setScanStatus(status);
  });
  const { resize } = useResizePlugin();
  // frameProcessor 僅在掃描中啟用
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      try {
        if (scanTrigger.value === 0) return;
        const model = modelRef.value;
        if (!model || labels.value.length === 0) return;
        const tensorData = resize(frame, {
          scale: { width: 416, height: 416 },
          pixelFormat: 'rgb',
          dataType: 'float32',
        });
        const outputshape = model.outputs[0].shape.slice(1);
        // const result = predictTflite(model, labels.value, tensorData, outputshape, 0.2, 0.45);
        // onInferenceResult(result);
        onScanStatus('掃描完成');
      } catch (e) {
        console.error('error:' + e);
      }
      scanTrigger.value = 0;
    },
    [trigger],
  );

  // 開始即時掃描
  const startScan = () => {
    setScanStatus('掃描中...');
    scanTrigger.value += 1;
    setTrigger(!trigger);
  };

  if (!device || !hasPermission) return <Text>相機初始化中...</Text>;

  return (
    <View style={styles.container}>
      <Camera style={StyleSheet.absoluteFill} device={device} isActive={true} frameProcessor={frameProcessor} />
      <ScrollView
        style={{ height: 300, borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.6)', marginBottom: 10, padding: 5 }}
      >
        {results.map((r, idx) => {
          const bb = r.box; // 假設是 [x, y, w, h]
          return (
            <View key={idx} style={{ marginBottom: 10, padding: 5, borderBottomWidth: 1, borderColor: '#ccc' }}>
              <Text style={styles.result}>{idx}</Text>
              <Text style={styles.result}>Object {idx + 1}:</Text>
              <Text style={styles.result}>Label: {r.label}</Text>
              <Text style={styles.result}>Confidence: {(r.prob * 100).toFixed(2)}%</Text>
            </View>
          );
        })}
      </ScrollView>
      <Button title="開始掃描" onPress={startScan} />
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
