import { useEffect, useState } from 'react';
import { ScrollView, Button, StyleSheet, Text, View } from 'react-native';
import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import { useSharedValue } from 'react-native-reanimated';
import { loadLabels, predictImageRN, preprocessLocalImage } from '../services/index';
export const TestTFLitePage = () => {
  const [results, setResults] = useState<{ label: string; prob: number; box?: number[] }[]>([]);
  const labels = useSharedValue<string[]>([]);
  const modelRef = useSharedValue<TensorflowModel | null>(null);

  useEffect(() => {
    (async () => {
      const lbls = await loadLabels();
      labels.value = lbls;
    })();
  }, []);
  useEffect(() => {
    (async () => {
      try {
        const loadedModel = await loadTensorflowModel({ url: 'model' });
        modelRef.value = loadedModel;
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);

  const TEST = async () => {
    const model = modelRef.value;
    if (!model || labels.value.length === 0) return;
    const tensorData = await preprocessLocalImage();
    const outputshape = model.outputs[0].shape.slice(1);
    const result = predictImageRN(model, labels.value, tensorData.data, outputshape, 0.1, 20);
    setResults(result);
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={{ height: 300, borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.6)', marginBottom: 10, padding: 5 }}
      >
        {results.map((r, idx) => {
          const bb = r.box; // 假設是 [x, y, w, h]
          return (
            <View key={idx} style={{ marginBottom: 10, padding: 5, borderBottomWidth: 1, borderColor: '#ccc' }}>
              <Text style={styles.result}>Object {idx + 1}:</Text>
              <Text style={styles.result}>Label: {r.label}</Text>
              <Text style={styles.result}>Confidence: {(r.prob * 100).toFixed(2)}%</Text>
            </View>
          );
        })}
      </ScrollView>
      <Button title="開始分析" onPress={TEST} />
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
