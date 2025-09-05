import { useEffect, useState } from 'react';
import { ScrollView, Button, StyleSheet, Text, View } from 'react-native';
import { loadLabels, preprocessAzureONNX, predictAzureOnnx, predictCustomOnnx, loadImage } from '../utils/index';
import { OutputData } from '../utils/types';
import { Canvas, Image, SkImage } from '@shopify/react-native-skia';
import { InferenceSession } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
export const OnnxPage = () => {
  const [results, setResults] = useState<{ label: string; prob: number; box: number[] }[]>([]);
  const [customs, setCustoms] = useState<OutputData | null>(null);
  const [image, setImage] = useState<SkImage | null>(null);
  const [azuremodel, setAzureModel] = useState<InferenceSession | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  useEffect(() => {
    (async () => {
      const lbls = await loadLabels();
      setLabels(lbls);
      const image = await loadImage();
      setImage(image);
    })();
  }, []);
  useEffect(() => {
    (async () => {
      try {
        const azuremodelPath = RNFS.DocumentDirectoryPath + '/onnxmodel.onnx';
        await RNFS.copyFileAssets('onnxmodel.onnx', azuremodelPath);
        const azureModel = await InferenceSession.create(azuremodelPath);
        setAzureModel(azureModel);
      } catch (err) {
        console.error('模型載入失敗', err);
      }
    })();
  }, []);

  const azureModel = async () => {
    setCustoms(null);
    if (!azuremodel || labels.length === 0) return;
    // 前處理
    const { tensor, origW, origH } = await preprocessAzureONNX(image);
    // 準備 feeds (使用模型 input 名稱)
    const inputName = azuremodel.inputNames[0]; // 或手動確認
    const feeds = { [inputName]: tensor };
    // 執行推理
    const result = await predictAzureOnnx(azuremodel, feeds, labels, origW, origH);
    setResults(result);
  };
  const customModel = async () => {
    setResults([]);
    const result = await predictCustomOnnx(image);
    setCustoms(result.data);
  };
  return (
    <View style={styles.container}>
      <Canvas style={{ flex: 1 }}>
        <Image image={image} fit="contain" x={0} y={0} width={256} height={256} />
      </Canvas>
      <ScrollView style={{ height: 300, borderWidth: 1, backgroundColor: 'rgba(0,0,0,0.6)', marginBottom: 10, padding: 5 }}>
        {results.length > 0 &&
          results.map((r, idx) => {
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
        {customs && (
          <View style={{ marginBottom: 10, padding: 5, borderBottomWidth: 1, borderColor: '#ccc' }}>
            <Text style={styles.result}>SYS:{customs.number['order1'] ?? 0}</Text>
            <Text style={styles.result}>DIA:{customs.number['order2'] ?? 0}</Text>
            <Text style={styles.result}>PUL:{customs.number['order3'] ?? 0}</Text>
            <Text style={styles.result}>Gentle:{customs.icon.gentle}</Text>
            <Text style={styles.result}>Text: {customs.text['label1'] ?? ''}</Text>
          </View>
        )}
      </ScrollView>
      <Button title="azure model分析" onPress={azureModel} />
      <Button title="custom model分析" onPress={customModel} />
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
