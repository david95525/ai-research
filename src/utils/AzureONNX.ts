import { Skia, ColorType, AlphaType, ImageInfo, SkImage, SkCanvas } from '@shopify/react-native-skia';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { validateYoloOutput } from './tesUtils';
function applyOrientation(canvas: SkCanvas, orientation: number, width: number, height: number) {
  switch (orientation) {
    case 2: // 水平翻轉
      canvas.scale(-1, 1);
      canvas.translate(-width, 0);
      break;
    case 3: // 旋轉 180
      canvas.rotate(180, width / 2, height / 2);
      break;
    case 4: // 垂直翻轉
      canvas.scale(1, -1);
      canvas.translate(0, -height);
      break;
    case 5: // 垂直翻轉 + 旋轉90
      canvas.rotate(90, width / 2, height / 2);
      canvas.scale(1, -1);
      break;
    case 6: // 旋轉 90
      canvas.rotate(90, width / 2, height / 2);
      break;
    case 7: // 水平翻轉 + 旋轉90
      canvas.rotate(90, width / 2, height / 2);
      canvas.scale(-1, 1);
      break;
    case 8: // 旋轉 270
      canvas.rotate(270, width / 2, height / 2);
      break;
    default:
      break; // orientation=1 → 不動
  }
}
/**
 * HWC -> CHW 轉換
 */
function hwcToChw(data: Float32Array, width: number, height: number): Float32Array {
  const chw = new Float32Array(3 * width * height);
  let offsetR = 0,
    offsetG = width * height,
    offsetB = 2 * width * height;
  let ptr = 0;
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      chw[offsetR++] = data[ptr++];
      chw[offsetG++] = data[ptr++];
      chw[offsetB++] = data[ptr++];
    }
  }
  return chw;
}
/**
 * React Native ONNX 前處理 (輸出 Tensor 可直接丟 session.run)
 */
export async function preprocessOnnx(
  image: SkImage | null,
  orientation: number = 1,
  targetSize: number = 416,
): Promise<{ tensor: Tensor; width: number; height: number; floatData: Float32Array; origW: number; origH: number }> {
  if (!image) throw new Error('Failed to load image');

  const origW = image.width();
  const origH = image.height();
  const scale = Math.min(targetSize / origW, targetSize / origH);

  const newW = Math.round(origW * scale);
  const newH = Math.round(origH * scale);

  const padW = targetSize - newW;
  const padH = targetSize - newH;
  const padLeft = Math.floor(padW / 2);
  const padTop = Math.floor(padH / 2);

  const surface = Skia.Surface.MakeOffscreen(targetSize, targetSize);
  if (!surface) throw new Error('Failed to create surface');
  const canvas = surface.getCanvas();
  const paint = Skia.Paint();

  // 填充黑色背景
  canvas.clear(Skia.Color('black'));

  // 修正方向
  applyOrientation(canvas, orientation, targetSize, targetSize);

  // 畫圖到 canvas 中間，保持比例
  canvas.drawImageRect(
    image,
    { x: 0, y: 0, width: origW, height: origH },
    { x: padLeft, y: padTop, width: newW, height: newH },
    paint,
  );

  // 取得像素 (RGBA)
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
    floatData[ptr++] = pixels[i + 2]; // B
    floatData[ptr++] = pixels[i + 1]; // G
    floatData[ptr++] = pixels[i]; // R
  }

  // HWC -> CHW
  const chwData = hwcToChw(floatData, targetSize, targetSize);

  // 建立 Tensor [1,3,H,W]
  const tensor = new Tensor('float32', chwData, [1, 3, targetSize, targetSize]);
  return { tensor, width: targetSize, height: targetSize, floatData, origW, origH };
}

/** Sigmoid */
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** 將 YOLO ONNX 輸出提取成 bounding boxes */
export const ONNXextractBB = (
  output: Float32Array | number[],
  outputShape: number[], // [H, W, C] e.g., [13, 13, 95]
  anchors: number[][], // [[0.573, 0.677], ...]
  probThreshold = 0.3,
) => {
  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  if (!Number.isInteger(numClass)) throw new Error(`Invalid output shape C=${channels}, numAnchor=${numAnchor}`);

  const arr: number[] = Array.isArray(output) ? output : Array.from(output);
  const results: { box: number[]; score: number[] }[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = a * (5 + numClass) + y * width * channels + x * channels;

        const tx = arr[base + 0];
        const ty = arr[base + 1];
        const tw = arr[base + 2];
        const th = arr[base + 3];
        const tobj = arr[base + 4];

        const cx = (sigmoid(tx) + x) / width;
        const cy = (sigmoid(ty) + y) / height;
        const w = Math.min((Math.exp(tw) * anchors[a][0]) / width, 1);
        const h = Math.min((Math.exp(th) * anchors[a][1]) / height, 1);

        const x1 = Math.max(0, cx - w / 2);
        const y1 = Math.max(0, cy - h / 2);
        const x2 = Math.min(1, cx + w / 2);
        const y2 = Math.min(1, cy + h / 2);

        // 過濾過小或過大框
        if (x2 - x1 < 0.01 || y2 - y1 < 0.01 || x2 - x1 > 0.95 || y2 - y1 > 0.95) continue;

        const objProb = sigmoid(tobj);

        // class logits -> softmax
        const classLogits: number[] = [];
        for (let k = 0; k < numClass; k++) classLogits.push(arr[base + 5 + k]);

        const maxLogit = Math.max(...classLogits);
        const expScores = classLogits.map((v) => Math.exp(v - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const classProbs = expScores.map((v) => (v / sumExp) * objProb);

        if (Math.max(...classProbs) > probThreshold) {
          results.push({ box: [x1, y1, x2 - x1, y2 - y1], score: classProbs });
        }
      }
    }
  }

  return results;
};

/** Non-Maximum Suppression */
export const nonMaxSuppression = (
  boxes: number[][],
  classProbs: number[][],
  scoreThreshold = 0.3,
  iouThreshold = 0.45,
  maxDetections = 100,
) => {
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
 * 將 YOLO 預測的 bounding boxes 從模型輸入尺寸轉回原始影像座標，並 clip 到影像邊界
 * @param boxes - [[x, y, w, h]] 以 targetSize 座標
 * @param origWidth - 原始影像寬
 * @param origHeight - 原始影像高
 * @param targetSize - 模型輸入尺寸 (default: 416)
 * @returns [[x, y, w, h]] 在原始影像座標
 */
export function mapBoxesToOriginal(
  boxes: number[][],
  origWidth: number,
  origHeight: number,
  targetSize = 416,
): number[][] {
  const scale = Math.min(targetSize / origWidth, targetSize / origHeight);
  const newW = origWidth * scale;
  const newH = origHeight * scale;
  const padLeft = (targetSize - newW) / 2;
  const padTop = (targetSize - newH) / 2;

  return boxes.map(([x, y, w, h]) => {
    const x1 = (x * targetSize - padLeft) / scale;
    const y1 = (y * targetSize - padTop) / scale;
    const w1 = (w * targetSize) / scale;
    const h1 = (h * targetSize) / scale;

    const finalX1 = Math.max(0, x1);
    const finalY1 = Math.max(0, y1);
    const finalX2 = Math.min(origWidth, finalX1 + w1);
    const finalY2 = Math.min(origHeight, finalY1 + h1);

    return [finalX1, finalY1, finalX2 - finalX1, finalY2 - finalY1];
  });
}
/** 篩選同 label 且重疊的 box，只保留最大機率 */
function filterBoxesPerLabelMax(boxes: number[][], labels: string[], probs: number[], iouThreshold: number = 0.3) {
  // 定義文字類 label
  const textLabels = new Set(['microlife', 'SYS', 'DIA', 'PUL']);

  // 依 prob 由高到低排序
  const candidates = boxes
    .map((b, i) => ({ box: b, label: labels[i], prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob);

  const finalBoxes: number[][] = [];
  const finalLabels: string[] = [];
  const finalProbs: number[] = [];

  const iou = (boxA: number[], boxB: number[]) => {
    const [ax, ay, aw, ah] = boxA;
    const [bx, by, bw, bh] = boxB;
    const x1 = Math.max(ax, bx);
    const y1 = Math.max(ay, by);
    const x2 = Math.min(ax + aw, bx + bw);
    const y2 = Math.min(ay + ah, by + bh);
    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const interArea = interW * interH;
    if (interArea === 0) return 0;
    const unionArea = aw * ah + bw * bh - interArea;
    return interArea / unionArea;
  };

  // 追蹤已經保留的文字 label
  const textLabelUsed = new Set<string>();

  for (const cand of candidates) {
    // 如果是文字類 label，只保留一個
    if (textLabels.has(cand.label) && textLabelUsed.has(cand.label)) continue;

    let overlapped = false;
    for (let j = 0; j < finalBoxes.length; j++) {
      if (cand.label === finalLabels[j] && iou(cand.box, finalBoxes[j]) > iouThreshold) {
        overlapped = true;
        break;
      }
    }

    if (!overlapped) {
      finalBoxes.push(cand.box);
      finalLabels.push(cand.label);
      finalProbs.push(cand.prob);
      if (textLabels.has(cand.label)) textLabelUsed.add(cand.label);
    }
  }

  return { finalBoxes, finalLabels, finalProbs };
}

/** ONNX 推理 + 後處理 */
export const predictOnnx = async (
  model: InferenceSession,
  feeds: { [x: string]: Tensor },
  labels: string[],
  origWidth: number,
  origHeight: number,
  targetSize = 416,
  probThreshold = 0.35,
  iouThreshold = 0.45,
  maxDetections = 100,
) => {
  try {
    const output = await model.run(feeds);
    const outputTensor: Tensor = output[model.outputNames[0]];
    const outputArray = new Float32Array(outputTensor.data as Float32Array);
    const [N, C, H, W] = outputTensor.dims;

    const ANCHORS = [
      [0.573, 0.677],
      [1.87, 2.06],
      [3.34, 5.47],
      [7.88, 3.53],
      [9.77, 9.17],
    ];

    const extracted = ONNXextractBB(Array.from(outputArray), [H, W, C], ANCHORS, probThreshold);
    if (!extracted || extracted.length === 0) return [];

    const boxes = extracted.map((r) => r.box);
    const classProbs = extracted.map((r) => r.score);
    const { selectedBoxes, selectedClasses, selectedProbs } = nonMaxSuppression(
      boxes,
      classProbs,
      probThreshold,
      iouThreshold,
      maxDetections,
    );
    const boxesOnOriginal = mapBoxesToOriginal(selectedBoxes, origWidth, origHeight, targetSize);
    // labels
    const selectedLabels = selectedClasses.map((i) => labels[i]);
    // per-label NMS 擴充
    const { finalBoxes, finalLabels, finalProbs } = filterBoxesPerLabelMax(
      boxesOnOriginal,
      selectedLabels,
      selectedProbs,
    );
    const results = finalBoxes.map((box, i) => ({
      label: finalLabels[i],
      prob: finalProbs[i],
      box,
    }));
    const resultorigin = boxesOnOriginal.map((box, i) => ({
      label: labels[selectedClasses[i]],
      prob: selectedProbs[i],
      box,
    }));
    console.log('origin results count:', resultorigin.length);
    console.log('Final results count:', results.length);
    return results;
  } catch (err) {
    console.error('predictOnnx error:', err);
    return [];
  }
};
