import { useEffect, useState } from 'react';
import { Button, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { Camera, useCameraDevice, useFrameProcessor } from 'react-native-vision-camera';
import { Worklets } from 'react-native-worklets-core';
import { useResizePlugin } from 'vision-camera-resize-plugin';

export const DemoTFLitePage = () => {
  const device = useCameraDevice('back');
  const [hasPermission, setHasPermission] = useState(false);
  const [results, setResults] = useState<
    {
      width: number | bigint;
      height: number | bigint;
      detection_scores: number | bigint;
      classes: number | bigint;
    }[]
  >([]);
  const [numdetections, setNumdetections] = useState<number | bigint>(0);
  const [scanStatus, setScanStatus] = useState('');
  const modelRef = useSharedValue<TensorflowModel | null>(null);
  const scanning = useSharedValue(true);

  // 相機權限與模型
  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === 'granted');
      try {
        const loadedModel = await loadTensorflowModel({ url: 'demo' });
        modelRef.value = loadedModel;
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);

  const onInferenceResult = Worklets.createRunOnJS((res: typeof results) => {
    setResults(res);
  });

  const onSetNumdetections = Worklets.createRunOnJS((num: number | bigint) => {
    setNumdetections(num);
  });
  const { resize } = useResizePlugin();
  // frameProcessor 僅在掃描中啟用
  const frameProcessor = useFrameProcessor(
    (frame) => {
      'worklet';
      try {
        if (!scanning.value) return;
        const model = modelRef.value;
        if (model == null) return;
        const resized = resize(frame, {
          scale: {
            width: 192,
            height: 192,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });
        const outputs = model.runSync([resized]);
        // 轉成陣列
        const detection_boxes = Object.values(outputs[0]);
        const detection_classes = Object.values(outputs[1]);
        const detection_scores = Object.values(outputs[2]);
        const num_detections = outputs[3];
        const selected: {
          width: number | bigint;
          height: number | bigint;
          detection_scores: number | bigint;
          classes: number | bigint;
        }[] = [];
        for (let i = 0; i < detection_boxes.length; i += 4) {
          const confidence = detection_scores[i / 4];
          const classes = detection_classes[i / 4];
          if (confidence > 0.5) {
            const left = detection_boxes[i];
            const top = detection_boxes[i + 1];
            const right = detection_boxes[i + 2];
            const bottom = detection_boxes[i + 3];
            const height = bottom - top;
            const width = right - left;
            selected.push({ height: height, width: width, detection_scores: confidence, classes: classes });
          }
        }
        onInferenceResult(selected);
        onSetNumdetections(num_detections[0]);
        scanning.value = false;
      } catch (e) {
        scanning.value = false;
        console.error(e);
      }
    },
    [modelRef],
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
        <Text style={styles.status}>Detected {numdetections} objects</Text>
        {results.map((r, idx) => (
          <Text key={idx} style={styles.result}>
            height:{r.height},width:{r.width},confidence:{r.detection_scores},classes:{r.classes}
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
