import { useEffect, useState } from 'react';
import { ScrollView, Button, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { loadLabels, predictTflite, preprocessTflite, loadImage } from '../utils/index';
import { Canvas, Image, SkImage } from '@shopify/react-native-skia';
export const TFLitePage = () => {
  const [results, setResults] = useState<{ label: string; prob: number; box: number[] }[]>([]);
  const [image, setImage] = useState<SkImage | null>(null);
  const labels = useSharedValue<string[]>([]);
  const modelRef = useSharedValue<TensorflowModel | null>(null);

  useEffect(() => {
    (async () => {
      const lbls = await loadLabels();
      labels.value = lbls;
      const image = await loadImage();
      setImage(image);
    })();
  }, []);
  useEffect(() => {
    (async () => {
      try {
        const loadedModel = await loadTensorflowModel({ url: 'tflitemodel' });
        modelRef.value = loadedModel;
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);

  const TEST = async () => {
    try {
      const model = modelRef.value;
      if (!model || labels.value.length === 0) return;
      const pre = await preprocessTflite(image);
      const outputshape = model.outputs[0].shape.slice(1);
      const result = predictTflite(model, labels.value, pre.data, outputshape, {
        width: pre.width,
        height: pre.height,
        origW: pre.origW,
        origH: pre.origH,
        scale: pre.scale,
        padLeft: pre.padLeft,
        padTop: pre.padTop,
      });
      setResults(result);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <View style={styles.container}>
      <Canvas style={{ flex: 1 }}>
        <Image image={image} fit="contain" x={0} y={0} width={256} height={256} />
      </Canvas>
      <ScrollView
        style={{ height: 300, borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.6)', marginBottom: 10, padding: 5 }}
      >
        {results.map((r, idx) => {
          const bb = r.box; // 假設是 [x, y, w, h]
          return (
            <View key={idx} style={{ marginBottom: 10, padding: 5, borderBottomWidth: 1, borderColor: '#ccc' }}>
              <Text style={styles.result}>{idx}</Text>
              <Text style={styles.result}>Label: {r.label}</Text>
              <Text style={styles.result}>Confidence: {(r.prob * 100).toFixed(2)}%</Text>
              <Text style={styles.result}>
                x:{r.box[0]}, y:{r.box[1]}, w:{r.box[2]}, h:{r.box[3]}
              </Text>
            </View>
          );
        })}
      </ScrollView>
      <Button title="tflite分析" onPress={TEST} />
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
