import { useEffect, useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { Camera, useCameraDevice, useFrameProcessor, PhotoFile } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { loadLabels, predictImageRN, caculateScale } from '../services/scanService';
import { useResizePlugin } from 'vision-camera-resize-plugin';

export const ScanTFLitePage = () => {
  const device = useCameraDevice('back');
  const [hasPermission, setHasPermission] = useState(false);
  const [results, setResults] = useState<{ label: string; prob: number; box?: number[] }[]>([]);
  const [scanStatus, setScanStatus] = useState('');
  const labels = useSharedValue<string[]>([]);
  const modelRef = useSharedValue<TensorflowModel | null>(null);
  const scanning = useSharedValue(true);

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
        const loadedModel = await loadTensorflowModel({ url: 'model' });
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
        if (!scanning.value) return;
        const model = modelRef.value;
        if (!model || labels.value.length === 0) return;

        const scale = caculateScale(frame);
        const tensorData = resize(frame, {
          scale: { width: 416, height: 416 },
          pixelFormat: 'rgb',
          dataType: 'float32',
        });
        const outputshape = model.outputs[0].shape.slice(1);
        const result = predictImageRN(model, labels.value, tensorData, outputshape, 0.1, 20);
        console.log(result);
        onInferenceResult(result);
        onScanStatus('掃描完成');
        scanning.value = false;
      } catch (e) {
        scanning.value = false;
        console.error(e);
      }
    },
    [labels, modelRef],
  );

  // 開始即時掃描
  const startScan = () => {
    setScanStatus('掃描中...');
    scanning.value = true;
  };

  if (!device || !hasPermission) return <Text>相機初始化中...</Text>;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true} // 相機畫面常顯示
        frameProcessor={frameProcessor}
      />
      <View style={styles.overlay}>
        <Text style={styles.status}>{scanStatus}</Text>
        {results.map((r, idx) => (
          <Text key={idx} style={styles.result}>
            {r.label} ({(r.prob * 100).toFixed(1)}%)
          </Text>
        ))}
      </View>
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
