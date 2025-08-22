// src/utils/ocr.ts
import TextRecognition from '@react-native-ml-kit/text-recognition';
import RNFS from 'react-native-fs';
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

const nonMaxSuppression = (boxes: number[][], scores: number[], iouThreshold = 0.45, maxDetections = 20) => {
  'worklet';
  const selected: number[] = [];
  const areas = boxes.map((b) => b[2] * b[3]);

  // 按 score 從大到小排序
  let idxs = scores
    .map((s, i) => i)
    .filter((i) => scores[i] > 0) // 過濾掉 0 或小於閾值的
    .sort((a, b) => scores[b] - scores[a]);
  while (selected.length < maxDetections && idxs.length > 0) {
    const i = idxs.shift()!;
    selected.push(i);
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
  return selected;
};
// --- 解析模型輸出 ---
const extractBB = (output: any, outputShape: number[], anchors: number[][]) => {
  'worklet';
  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  const boxes: number[][] = [];
  const classProbs: number[][] = [];
  // 數值穩定 sigmoid
  const sigmoid = (x: number) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

  // 取值 function，支援 array 或 { "0": val } 物件
  const getOutput = (idx: number) => (Array.isArray(output) ? output[idx] : output[idx.toString()]);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = ((y * width + x) * numAnchor + a) * (numClass + 5);

        // x, y center (加上格子索引，並歸一化)
        const ox = (sigmoid(getOutput(base + 0)) + x) / width;
        const oy = (sigmoid(getOutput(base + 1)) + y) / height;

        // w, h (乘上 anchor，並歸一化)
        const ow = (Math.exp(getOutput(base + 2)) * anchors[a][0]) / width;
        const oh = (Math.exp(getOutput(base + 3)) * anchors[a][1]) / height;

        // objectness
        const obj = sigmoid(getOutput(base + 4));

        // class probabilities (softmax)
        const clsScores: number[] = [];
        for (let c = 0; c < numClass; c++) {
          clsScores.push(getOutput(base + 5 + c));
        }

        const maxScore = Math.max(...clsScores);
        const expScores = clsScores.map((v) => Math.exp(v - maxScore));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const probs = expScores.map((v) => (v / sumExp) * obj);

        // convert center to top-left
        boxes.push([ox - ow / 2, oy - oh / 2, ow, oh]);
        classProbs.push(probs);
      }
    }
  }
  return { boxes, classProbs };
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

  const { boxes, classProbs } = extractBB(output, outputShape, ANCHORS);

  const maxProbs = classProbs.map((c: number[]) => Math.max(...c)); //每個 box 的最大 class 概率
  let validIdx = maxProbs.map((p, i) => (p > probThreshold ? i : -1)).filter((i) => i >= 0); //過濾出概率大於閾值的索引
  validIdx = validIdx.sort((a, b) => maxProbs[b] - maxProbs[a]); //按概率從大到小排序
  //取得對應的 boxes 和最大概率
  const filteredBoxes = validIdx.map((i) => boxes[i]);
  const filteredProbs = validIdx.map((i) => maxProbs[i]);

  const selectedBoxesIdx = nonMaxSuppression(filteredBoxes, filteredProbs, maxDetections);
  return selectedBoxesIdx.map((i) => ({
    label: labels[0], // 單類別
    prob: filteredProbs[i], // 對應的概率
    box: filteredBoxes[i], // 對應的 box
  }));
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
