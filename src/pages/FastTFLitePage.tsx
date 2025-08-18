import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View } from "react-native";
import { loadTensorflowModel } from "react-native-fast-tflite";
import { Camera, useCameraDevice, useFrameProcessor } from "react-native-vision-camera";
import { useSharedValue, useAnimatedReaction, runOnJS } from "react-native-reanimated";
import { useResizePlugin } from "vision-camera-resize-plugin";

let model: Awaited<ReturnType<typeof loadTensorflowModel>> | null = null;

export const FastTFLitePage = () => {
  const device = useCameraDevice("back");
  const [hasPermission, setHasPermission] = useState(false);
  const [resultText, setResultText] = useState("");

  const lastProcessed = useSharedValue(0);
  const result = useSharedValue(""); // 儲存推論結果

  const { resize } = useResizePlugin();

  // 請求相機權限 + 載入模型
  useEffect(() => {
    (async () => {
      const cameraPermission = await Camera.requestCameraPermission();
      setHasPermission(cameraPermission === "granted");

      model = await loadTensorflowModel({
        url: "../assets/model.tflite", 
      });
      console.log("模型載入完成");
    })();
  }, []);

  // 把 sharedValue 結果同步到 React state（跨 thread 用 runOnJS）
  useAnimatedReaction(
    () => result.value,
    (value) => {
      runOnJS(setResultText)(value);
    }
  );

  // Frame Processor
  const frameProcessor = useFrameProcessor((frame) => {
    "worklet";
    if (!model) return;

    const now = Date.now();
    if (now - lastProcessed.value < 1000) return; // 限制 1FPS
    lastProcessed.value = now;

    // 影像 resize
    const input = resize(frame, {
      scale: { width: 192, height: 192 },
      pixelFormat: "rgb",
      dataType: "float32",
    });

    // normalize
    for (let i = 0; i < input.length; i++) {
      input[i] /= 255.0;
    }

    // 推論（回到 JS thread）
    model.run([input])
      .then((output) => {
        const probs = output[0] as Float32Array;
        let maxIdx = 0;
        let maxVal = -Infinity;
        for (let i = 0; i < probs.length; i++) {
          if (probs[i] > maxVal) {
            maxVal = probs[i];
            maxIdx = i;
          }
        }
        const percent = (maxVal * 100).toFixed(1);
        result.value = `分類結果：類別 ${maxIdx}，信心 ${percent}%`;
      })
      .catch((e) => {
        console.error("推論錯誤", e);
      });

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
        fps={5} // 降低負載
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
