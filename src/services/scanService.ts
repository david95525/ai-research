// src/utils/ocr.ts
import TextRecognition from '@react-native-ml-kit/text-recognition';
import RNFS from 'react-native-fs';
import { Image } from 'react-native';
import ImageResizer from 'react-native-image-resizer';
import { Buffer } from 'buffer';
/**
 * 對指定圖片路徑執行 OCR，回傳所有文字
 */
export async function runOCR(photo: { path: string }): Promise<string> {
  try {
    const result = await TextRecognition.recognize(`file://${photo.path}`);
    return result.blocks.map((block) => block.text).join('\n');
  } catch (err) {
    console.error('OCR failed:', err);
    throw err;
  }
}

/**
 * 載入 labels.txt，回傳每一行的陣列
 */
export async function loadLabels(): Promise<string[]> {
  try {
    // readFileAssets 只要檔名，不需要 path
    const content = await RNFS.readFileAssets('labels.txt', 'utf8');
    const lines = content
      .split('\n')
      .map((l) => l.trim())
      .filter(Boolean);
    return lines;
  } catch (err) {
    console.error('Labels load failed:', err);
    return [];
  }
}

// --- 解析模型輸出 ---
const extractBB = (output: any, outputShape: number[], anchors: number[][]) => {
  'worklet';
  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  const boxes: number[][] = [];
  const classProbs: number[][] = [];

  // sigmoid 避免溢出
  const sigmoid = (x: number) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

  // 轉成 array
  const arr: number[] = Array.isArray(output) ? output : Object.values(output);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = ((y * width + x) * numAnchor + a) * (numClass + 5);

        // x, y center (加上格子索引並歸一化)
        const ox = (sigmoid(arr[base + 0]) + x) / width;
        const oy = (sigmoid(arr[base + 1]) + y) / height;

        // w, h (乘 anchor 並歸一化)
        const ow = (Math.exp(arr[base + 2]) * anchors[a][0]) / width;
        const oh = (Math.exp(arr[base + 3]) * anchors[a][1]) / height;

        // objectness (目前不用乘，只保留)
        const obj = sigmoid(arr[base + 4]);
        // class logits → softmax
        const clsScores: number[] = [];
        for (let c = 0; c < numClass; c++) {
          clsScores.push(arr[base + 5 + c]);
        }

        const maxScore = Math.max(...clsScores);
        const expScores = clsScores.map((v) => Math.exp(v - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map((v) => v / sumExp);
        const probsWithObj = probs.map((v) => v * obj);
        // 存結果
        boxes.push([ox - ow / 2, oy - oh / 2, ow, oh]);
        classProbs.push(probsWithObj);
      }
    }
  }
  return { boxes, classProbs };
};
// 改寫 NMS
const nonMaxSuppression = (boxes: number[][], maxProbs: number[], iouThreshold = 0.45, maxDetections = 20) => {
  const selectedBoxes: number[][] = [];
  const selectedIndices: number[] = [];
  const selectedProbs: number[] = [];

  const areas = boxes.map((b) => b[2] * b[3]);

  let idxs = maxProbs
    .map((_, i) => i)
    .filter((i) => maxProbs[i] > 0)
    .sort((a, b) => maxProbs[b] - maxProbs[a]);

  while (selectedBoxes.length < maxDetections && idxs.length > 0) {
    const i = idxs.shift()!;
    selectedBoxes.push(boxes[i]);
    selectedIndices.push(i); // 直接存索引
    selectedProbs.push(maxProbs[i]);

    for (let j = idxs.length - 1; j >= 0; j--) {
      const [x1, y1, w1, h1] = boxes[i];
      const [x2, y2, w2, h2] = boxes[idxs[j]];

      const interW = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
      const interH = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
      const interArea = interW * interH;
      const iou = interArea / (areas[i] + areas[idxs[j]] - interArea);

      if (iou > iouThreshold) idxs.splice(j, 1);
    }
  }

  return { selectedBoxes, selectedIndices, selectedProbs };
};
const extractBBFiltered = (output: any, outputShape: number[], anchors: number[][], probThreshold = 0.1) => {
  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  const boxes: number[][] = [];
  const classProbs: number[][] = [];
  const objectnessArr: number[] = [];

  const sigmoid = (x: number) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
  const arr: number[] = Array.isArray(output) ? output : Array.from(output);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = ((y * width + x) * numAnchor + a) * (numClass + 5);

        // x, y center
        const ox = (sigmoid(arr[base + 0]) + x) / width;
        const oy = (sigmoid(arr[base + 1]) + y) / height;

        // w, h
        const ow = (Math.exp(arr[base + 2]) * anchors[a][0]) / width;
        const oh = (Math.exp(arr[base + 3]) * anchors[a][1]) / height;

        // objectness
        const obj = sigmoid(arr[base + 4]);
        objectnessArr.push(obj);

        // class logits → softmax
        const clsScores: number[] = [];
        for (let c = 0; c < numClass; c++) clsScores.push(arr[base + 5 + c]);

        const maxScore = Math.max(...clsScores);
        const expScores = clsScores.map((v) => Math.exp(v - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map((v) => v / sumExp);

        // 存結果
        boxes.push([ox - ow / 2, oy - oh / 2, ow, oh]);
        classProbs.push(probs);
      }
    }
  }

  // 先用 softmax 篩選 box，不乘 objectness
  const maxClassInfo = boxes.map((_, i) => {
    const probs = classProbs[i];
    const maxProb = Math.max(...probs);
    const classIdx = probs.indexOf(maxProb);
    return { maxProb, classIdx, obj: objectnessArr[i] };
  });

  const validIdx = maxClassInfo.map((info, i) => (info.maxProb > probThreshold ? i : -1)).filter((i) => i >= 0);

  // 可以在這裡乘 objectness 來做排序或後續 NMS
  validIdx.sort((a, b) => {
    // 排序依最大概率乘 objectness
    const scoreA = maxClassInfo[a].maxProb * maxClassInfo[a].obj;
    const scoreB = maxClassInfo[b].maxProb * maxClassInfo[b].obj;
    return scoreB - scoreA;
  });

  return { boxes, classProbs, objectnessArr, validIdx, maxClassInfo };
};

// --- 核心推理函式 ---
export const predictImageRN = (
  model: any,
  labels: string[],
  inputImage: Float32Array,
  outputShape: number[],
  probThreshold: number = 0.1,
  maxDetections: number = 20,
) => {
  'worklet';
  // --- 模型推理 ---
  const inputTensor = inputImage; // 已經是 Float32Array

  const output: any = model.runSync([inputTensor])[0]; // output: [H,W,C]

  // --- 後處理 ---
  const ANCHORS = [
    [0.573, 0.677],
    [1.87, 2.06],
    [3.34, 5.47],
    [7.88, 3.53],
    [9.77, 9.17],
  ];

  const { boxes, classProbs, objectnessArr, validIdx, maxClassInfo } = extractBBFiltered(
    output,
    outputShape,
    ANCHORS,
    probThreshold,
  );

  // validIdx 已經包含 threshold 篩選過的 box index
  // 並且可用 objectness 進行排序

  const filteredBoxes = validIdx.map((i) => boxes[i]);
  const filteredMaxProbs = validIdx.map((i) => maxClassInfo[i].maxProb);
  const filteredClassIdx = validIdx.map((i) => maxClassInfo[i].classIdx);
  console.log(filteredClassIdx.slice(0, 20));
  // NMS 只傳 boxes + maxProbs
  const { selectedBoxes, selectedIndices, selectedProbs } = nonMaxSuppression(filteredBoxes, filteredMaxProbs);

  // selectedIndices 對應 filteredClassIdx
  const selectedClasses = selectedIndices.map((i) => filteredClassIdx[i]);
  const results = selectedBoxes.map((box, i) => ({
    label: labels[selectedClasses[i]], // 對應的類別文字
    prob: selectedProbs[i], // 對應的最大概率
    box: box, // 對應的 box
  }));

  return results;
};
interface scale {
  width: number;
  height: number;
}
/**
 * - 保持原始比例
 * - 寬高對齊 32 的倍數
 * - 輸出 scale
 */
export const caculateScale = (frame: any, defaultInputSize: number = 416 * 416): scale => {
  'worklet';
  // 計算新的比例
  const ratio = Math.sqrt(defaultInputSize / (frame.width * frame.height));
  let newWidth = Math.ceil(frame.width * ratio);
  let newHeight = Math.ceil(frame.height * ratio);

  // 對齊 32 倍數
  newWidth = 32 * Math.ceil(newWidth / 32);
  newHeight = 32 * Math.ceil(newHeight / 32);
  return { width: newWidth, height: newHeight };
};
const DEFAULT_INPUT_SIZE = 416;
export const copyAssetToDocDir = async () => {
  try {
    const destPath = RNFS.DocumentDirectoryPath + '/scanbp.png';
    const base64 = await RNFS.readFileAssets('scanbp.png', 'base64');
    await RNFS.writeFile(destPath, base64, 'base64');
    return destPath;
  } catch (err) {
    console.error('copyAssetToDocDir failed:', err);
    return '';
  }
};
/**
 * 1. 讀取本地 PNG 並做前處理（縮放 + 對齊 32 的倍數）
 * 回傳 Float32Array，值 normalized 到 [0,1]
 */
export async function preprocessLocalImage(localPath: string): Promise<Float32Array> {
  try {
    //嘗試用 Image.resolveAssetSource 拿原始寬高（專案 assets）
    let width = 0;
    let height = 0;

    const source = Image.resolveAssetSource({ uri: 'file://' + localPath });
    width = source.width;
    height = source.height;

    // fallback: 假設我們先縮放到 DEFAULT_INPUT_SIZE 再拿尺寸
    width = DEFAULT_INPUT_SIZE;
    height = DEFAULT_INPUT_SIZE;

    //計算縮放比例
    const ratio = Math.sqrt(DEFAULT_INPUT_SIZE / (width * height));
    let newWidth = Math.ceil(width * ratio);
    let newHeight = Math.ceil(height * ratio);

    // 對齊到 32 的倍數
    newWidth = 32 * Math.ceil(newWidth / 32);
    newHeight = 32 * Math.ceil(newHeight / 32);

    //用 ImageResizer 縮放圖片
    const resized = await ImageResizer.createResizedImage(localPath, newWidth, newHeight, 'PNG', 100);

    //讀取縮放後檔案為 base64
    const base64 = await RNFS.readFile(resized.uri, 'base64');

    //轉 Float32Array (RGB normalized)
    const binary = Buffer.from(base64, 'base64');
    const floatData = new Float32Array(newWidth * newHeight * 3);
    let ptr = 0;

    for (let i = 0; i < binary.length; i += 4) {
      // PNG 是 RGBA
      floatData[ptr++] = binary[i] / 255; // R
      floatData[ptr++] = binary[i + 1] / 255; // G
      floatData[ptr++] = binary[i + 2] / 255; // B
      // 忽略 A
    }
    return floatData;
  } catch (err) {
    console.error('preprocessLocalImage failed:', err);
    return new Float32Array(0);
  }
}
