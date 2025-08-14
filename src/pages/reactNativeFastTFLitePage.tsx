import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { runOnJS } from "react-native-reanimated";
import {
  Camera,
  useCameraDevice,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useSharedValue } from "react-native-worklets-core";
import { useResizePlugin } from "vision-camera-resize-plugin";

let model: Awaited<ReturnType<typeof loadTensorflowModel>> | null = null;

export const ReactNativeFastTFLitePage = () => {
  const device = useCameraDevice("back");
  const [hasPermission, setHasPermission] = useState(false);
  const [resultText, setResultText] = useState("");
  const lastProcessed = useSharedValue(0);

  const { resize } = useResizePlugin();

  // 相機權限 + 載入模型
  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === "granted");

      model = await loadTensorflowModel({ url: "../assets/model.tflite" });
      console.log("模型載入完成");
    })();
  }, []);

  const processFrame = async (buffer: Uint8Array) => {
    if (!model) return;

    try {
      const input = new Float32Array(buffer.length);
      for (let i = 0; i < buffer.length; i++) {
        input[i] = buffer[i] / 255.0;
      }

      const output = await model.run([input]); // 注意：是陣列包 TypedArray

      const probs = output[0] as Float32Array;
      const maxIdx = probs.indexOf(Math.max(...probs));
      const percent = (probs[maxIdx] * 100).toFixed(1);
      setResultText(`分類結果：類別 ${maxIdx}，信心 ${percent}%`);
    } catch (e) {
      console.error("推論錯誤", e);
    }
  };

  const frameProcessor = useFrameProcessor((frame) => {
    "worklet";
    const now = Date.now();
    if (now - lastProcessed.value < 1000) return;

    lastProcessed.value = now;

    // 官方用法，resize 傳入設定物件
    const resized = resize(frame, {
      scale: {
        width: 192,
        height: 192,
      },
      pixelFormat: "rgb",
      dataType: "uint8",
    });

    runOnJS(processFrame)(resized);
  }, []);

  if (!device || !hasPermission) {
    return <Text>相機初始化中...</Text>;
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        fps={5} // 注意是 fps，不是 frameProcessorFps
      />
      <View style={styles.overlay}>
        <Text style={styles.result}>{resultText}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1 },
  overlay: {
    position: "absolute",
    bottom: 0,
    width: "100%",
    padding: 16,
    backgroundColor: "rgba(0,0,0,0.6)",
  },
  result: {
    color: "#fff",
    fontSize: 18,
  },
});
