import { Skia, ColorType, AlphaType, ImageInfo, SkImage } from '@shopify/react-native-skia';
import { validateYoloOutput, validatePreprocess } from './tesUtils';
/**
 * Preprocess image for TFLite (letterbox + padding)
 * 回傳 Float32Array，值 normalized 到 [0,1]
 */
export async function preprocessTflite(
  image: SkImage | null,
  targetSize = 416,
): Promise<{
  data: Float32Array;
  width: number; // 模型輸入寬度
  height: number; // 模型輸入高度
  origW: number; // 原始圖寬
  origH: number; // 原始圖高
  scale: number; // 縮放比例
  padLeft: number; // 左邊 padding
  padTop: number; // 上邊 padding
}> {
  if (!image) throw new Error('Failed to load image');

  const origW = image.width();
  const origH = image.height();

  // 計算 letterbox scale
  const scale = Math.min(targetSize / origW, targetSize / origH);
  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);

  const padW = targetSize - newW;
  const padH = targetSize - newH;
  const padLeft = Math.floor(padW / 2);
  const padTop = Math.floor(padH / 2);

  // 建立 surface
  const surface = Skia.Surface.MakeOffscreen(targetSize, targetSize);
  if (!surface) throw new Error('Failed to create surface');
  const canvas = surface.getCanvas();
  const paint = Skia.Paint();

  // 填充黑色背景
  canvas.clear(Skia.Color('black'));

  // 畫圖到 canvas 中間，保持比例
  canvas.drawImageRect(
    image,
    { x: 0, y: 0, width: origW, height: origH },
    { x: padLeft, y: padTop, width: newW, height: newH },
    paint,
  );

  // 拿到像素 (RGBA Uint8Array)
  const snapshot = surface.makeImageSnapshot();
  const imageInfo: ImageInfo = {
    width: targetSize,
    height: targetSize,
    colorType: ColorType.RGBA_8888,
    alphaType: AlphaType.Unpremul,
  };
  const pixels = snapshot.readPixels(0, 0, imageInfo);
  if (!pixels) throw new Error('Failed to read pixels');

  // Uint8 -> Float32 (RGB)
  const floatData = new Float32Array(targetSize * targetSize * 3);
  let ptr = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    floatData[ptr++] = pixels[i] / 255; // R
    floatData[ptr++] = pixels[i + 1] / 255; // G
    floatData[ptr++] = pixels[i + 2] / 255; // B
  }
  // 避免展開造成爆 stack
  let minVal = Infinity,
    maxVal = -Infinity;
  for (let v of floatData) {
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  return { data: floatData, width: targetSize, height: targetSize, origW, origH, scale, padLeft, padTop };
}
/**
 * 解析模型輸出
 */
const TFLiteextractBB = (
  output: Float32Array | number[],
  outputShape: number[], // [H, W, C] e.g. [13,13,95]
  anchors: number[][], // 5 anchors
  probThreshold = 0.3,
) => {
  'worklet';

  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  if (!Number.isInteger(numClass)) {
    throw new Error(`Invalid output shape, got C=${channels}, A=${numAnchor}`);
  }
  const stride = 5 + numClass;

  const sigmoid = (x: number) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));

  const arr: number[] = Array.isArray(output) ? output : Array.from(output);
  const results: { box: number[]; score: number[] }[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = ((y * width + x) * stride + 0) * numAnchor + a;
        const tx = arr[base + 0];
        const ty = arr[base + 1];
        const tw = arr[base + 2];
        const th = arr[base + 3];
        const tobj = arr[base + 4];

        const cx = (sigmoid(tx) + x) / width;
        const cy = (sigmoid(ty) + y) / height;
        const w = (Math.exp(tw) * anchors[a][0]) / width; // width = 13
        const h = (Math.exp(th) * anchors[a][1]) / height; // height = 13
        const box = [cx - w / 2, cy - h / 2, w, h];
        const obj = sigmoid(tobj);

        // class logits
        const classLogits: number[] = [];
        for (let k = 0; k < numClass; k++) {
          classLogits.push(arr[base + 5 + k]);
        }
        // softmax
        const maxLogit = Math.max(...classLogits);
        const expScores = classLogits.map((v) => Math.exp(v - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const classProbs = expScores.map((v) => (v / sumExp) * obj);
        if (Math.max(...classProbs) > probThreshold) {
          results.push({ box, score: classProbs });
        }
      }
    }
  }
  return results;
};

/**
 * NMS
 */
const nonMaxSuppression = (
  boxes: number[][],
  classProbs: number[][],
  scoreThreshold: number,
  iouThreshold: number,
  maxDetections: number,
) => {
  'worklet';
  const selectedBoxes: number[][] = [];
  const selectedClasses: number[] = [];
  const selectedProbs: number[] = [];

  const maxProbs = classProbs.map((p) => Math.max(...p));
  const maxClasses = classProbs.map((p) => p.indexOf(Math.max(...p)));

  const boxesCopy = boxes.slice();
  const probsCopy = classProbs.map((p) => p.slice());
  const maxProbsCopy = maxProbs.slice();
  const maxClassesCopy = maxClasses.slice();

  while (selectedBoxes.length < maxDetections && boxesCopy.length > 0) {
    let i = maxProbsCopy.indexOf(Math.max(...maxProbsCopy));
    if (maxProbsCopy[i] < scoreThreshold) break;

    selectedBoxes.push(boxesCopy[i]);
    selectedClasses.push(maxClassesCopy[i]);
    selectedProbs.push(maxProbsCopy[i]);

    const [x1, y1, w1, h1] = boxesCopy[i];

    for (let j = 0; j < boxesCopy.length; j++) {
      if (j === i) continue;
      const [x2, y2, w2, h2] = boxesCopy[j];
      const interW = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
      const interH = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
      const interArea = interW * interH;
      const unionArea = w1 * h1 + w2 * h2 - interArea;
      const iou = interArea / unionArea;

      if (iou > iouThreshold) {
        probsCopy[j][maxClassesCopy[i]] = 0;
        maxProbsCopy[j] = Math.max(...probsCopy[j]);
        maxClassesCopy[j] = probsCopy[j].indexOf(maxProbsCopy[j]);
      }
    }

    boxesCopy.splice(i, 1);
    probsCopy.splice(i, 1);
    maxProbsCopy.splice(i, 1);
    maxClassesCopy.splice(i, 1);
  }

  return { selectedBoxes, selectedClasses, selectedProbs };
};
/**
 * 將 YOLO 模型輸出還原到原圖座標
 */
const restoreBoxToOriginal = (
  box: number[], // [x_min, y_min, w, h] normalized，相對模型輸入大小
  width: number, // 模型輸入寬度
  height: number, // 模型輸入高度
  scale: number,
  padLeft: number,
  padTop: number,
  origW: number,
  origH: number,
) => {
  // 還原到模型輸入像素座標
  const x_min = box[0] * width;
  const y_min = box[1] * height;
  const w = box[2] * width;
  const h = box[3] * height;

  // 去掉 padding，再還原縮放到原圖
  const x1 = Math.max(0, (x_min - padLeft) / scale);
  const y1 = Math.max(0, (y_min - padTop) / scale);
  const x2 = Math.min(origW, (x_min + w - padLeft) / scale);
  const y2 = Math.min(origH, (y_min + h - padTop) / scale);

  return [x1, y1, x2, y2]; // 原圖座標
};

/**
 * 核心推理函式 (改寫)
 */
export const predictTflite = (
  model: any,
  labels: string[],
  inputImage: Float32Array,
  outputShape: number[],
  preprocessInfo: {
    width: number;
    height: number;
    origW: number;
    origH: number;
    scale: number;
    padLeft: number;
    padTop: number;
  },
  probThreshold = 0.1,
  iouThreshold = 0.45,
  maxDetections = 50,
) => {
  'worklet';
  try {
    const output: any = model.runSync([inputImage])[0]; // [[HxWxC]]
    const ANCHORS = [
      [0.573, 0.677],
      [1.87, 2.06],
      [3.34, 5.47],
      [7.88, 3.53],
      [9.77, 9.17],
    ];

    const [H, W, C] = outputShape;
    const extractResults = TFLiteextractBB(output, outputShape, ANCHORS, probThreshold);
    if (extractResults.length === 0) return [];

    const boxes = extractResults.map((r) => r.box);
    const classProbs = extractResults.map((r) => r.score);

    const { selectedBoxes, selectedClasses, selectedProbs } = nonMaxSuppression(
      boxes,
      classProbs,
      probThreshold,
      iouThreshold,
      maxDetections,
    );

    const results = selectedBoxes.map((box, i) => {
      const restored = restoreBoxToOriginal(
        box,
        preprocessInfo.width,
        preprocessInfo.height,
        preprocessInfo.scale,
        preprocessInfo.padLeft,
        preprocessInfo.padTop,
        preprocessInfo.origW,
        preprocessInfo.origH,
      );

      return {
        label: labels[selectedClasses[i]],
        prob: selectedProbs[i],
        box: restored, // [x1, y1, x2, y2] 原圖座標
      };
    });
    console.log('Final results count:', results.length);
    console.log('Sample results:', results.slice(0, results.length));
    return results;
  } catch (err) {
    console.error('predicttflite error:', err);
    return [
      {
        label: '',
        prob: 0,
        box: [0],
      },
    ];
  }
};
