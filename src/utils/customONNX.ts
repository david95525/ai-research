import { Skia, ColorType, AlphaType, SkImage } from '@shopify/react-native-skia';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { applyOrientation, hwcToChw } from './commonUtils';
import TextRecognition from '@react-native-ml-kit/text-recognition';
import { writeFile, DocumentDirectoryPath } from 'react-native-fs';
import RNFS from 'react-native-fs';
import { fromByteArray } from 'base64-js';
/**
 * React Native ONNX 完整 Custom model 前處理
 */
async function preprocessCustomONNX(
  image: SkImage | null,
  orientation: number = 1,
  targetSize: number = 640,
  stride: number = 8,
  auto: boolean = true,
  scaleFill: boolean = false,
  scaleup: boolean = true,
): Promise<{ tensor: Tensor; width: number; height: number; floatData: Float32Array; origW: number; origH: number }> {
  if (!image) throw new Error('Failed to load image');

  const origW = image.width();
  const origH = image.height();

  // -------------------
  // 計算縮放比例
  // -------------------
  let r = Math.min(targetSize / origW, targetSize / origH);
  if (!scaleup) r = Math.min(r, 1.0);

  let newW = Math.round(origW * r);
  let newH = Math.round(origH * r);

  let padW = targetSize - newW;
  let padH = targetSize - newH;

  // scaleFill: 不保留比例，直接拉伸到目標尺寸
  if (scaleFill) {
    newW = targetSize;
    newH = targetSize;
    r = [targetSize / origW, targetSize / origH] as unknown as number;
    padW = 0;
    padH = 0;
  } else {
    // auto: 確保 padding 為 stride 的倍數
    if (auto) {
      padW = padW % stride;
      padH = padH % stride;
    }
  }

  const padLeft = Math.floor(padW / 2);
  const padTop = Math.floor(padH / 2);

  // -------------------
  // 建立 Canvas
  // -------------------
  const surface = Skia.Surface.MakeOffscreen(targetSize, targetSize);
  if (!surface) throw new Error('Failed to create surface');

  const canvas = surface.getCanvas();
  const paint = Skia.Paint();

  // 背景填充灰色 (114,114,114)
  canvas.clear(Skia.Color('#727272'));

  // 修正方向
  applyOrientation(canvas, orientation, targetSize, targetSize);

  // 繪製圖片到 canvas 中心
  canvas.drawImageRect(image, { x: 0, y: 0, width: origW, height: origH }, { x: padLeft, y: padTop, width: newW, height: newH }, paint);

  // -------------------
  // 取得像素
  // -------------------
  const snapshot = surface.makeImageSnapshot();
  const imageInfo = {
    width: targetSize,
    height: targetSize,
    colorType: ColorType.RGBA_8888,
    alphaType: AlphaType.Unpremul,
  };
  const pixels = snapshot.readPixels(0, 0, imageInfo);
  if (!pixels) throw new Error('Failed to read pixels');

  // Uint8 -> Float32 (RGB) 並 normalize
  const floatData = new Float32Array(targetSize * targetSize * 3);
  let ptr = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    floatData[ptr++] = pixels[i + 2] / 255.0; // B
    floatData[ptr++] = pixels[i + 1] / 255.0; // G
    floatData[ptr++] = pixels[i] / 255.0; // R
  }

  // HWC -> CHW
  const chwData = hwcToChw(floatData, targetSize, targetSize);

  // 建立 Tensor [1,3,H,W]
  const tensor = new Tensor('float32', chwData, [1, 3, targetSize, targetSize]);

  return { tensor, width: targetSize, height: targetSize, floatData, origW, origH };
}
// group boxes by Y-level
type Box = { bbox: number[]; label: string; confidence: number };
function groupByYLevel(boxes: Box[], threshold = 10): Box[][] {
  const sorted = [...boxes].sort((a, b) => a.bbox[1] - b.bbox[1]);
  const grouped: Box[][] = [];
  let currentGroup: Box[] = [];
  let prevY: number | null = null;

  for (const box of sorted) {
    const y1 = box.bbox[1];
    if (prevY === null || Math.abs(y1 - prevY) <= threshold) {
      currentGroup.push(box); // ✅ 這裡沒問題了
    } else {
      grouped.push(currentGroup);
      currentGroup = [box]; // ✅ 同樣沒問題
    }
    prevY = y1;
  }
  if (currentGroup.length) grouped.push(currentGroup);
  return grouped;
}
// 將 bbox 從 targetSize 映射回原圖大小
function scaleCoords(
  coords: number[][],
  img1Shape: [number, number], // [height, width] of model input
  img0Shape: [number, number], // [origH, origW]
  ratioPad?: [[number, number], [number, number]], // optional precomputed [gain, [padW, padH]]
): number[][] {
  let gain: number;
  let pad: [number, number];

  if (!ratioPad) {
    gain = Math.min(img1Shape[0] / img0Shape[0], img1Shape[1] / img0Shape[1]);
    pad = [(img1Shape[1] - img0Shape[1] * gain) / 2, (img1Shape[0] - img0Shape[0] * gain) / 2];
  } else {
    gain = ratioPad[0][0];
    pad = ratioPad[1];
  }

  return coords.map(([x1, y1, x2, y2]) => {
    let nx1 = (x1 - pad[0]) / gain;
    let ny1 = (y1 - pad[1]) / gain;
    let nx2 = (x2 - pad[0]) / gain;
    let ny2 = (y2 - pad[1]) / gain;
    // clip to image size
    nx1 = Math.max(Math.min(nx1, img0Shape[1]), 0);
    ny1 = Math.max(Math.min(ny1, img0Shape[0]), 0);
    nx2 = Math.max(Math.min(nx2, img0Shape[1]), 0);
    ny2 = Math.max(Math.min(ny2, img0Shape[0]), 0);
    return [nx1, ny1, nx2, ny2];
  });
}

// 檢查是否重疊
function isOverlap(newBox: number[], existingBoxes: number[][], threshold = 0.5): boolean {
  return existingBoxes.some((b) => iou(newBox, b) > threshold);
}

// 將 label 轉成 ICON key
function mapIconLabel(label: string): [string, number] {
  if (label === 'IHB') return ['ihb', 1];
  if (label === 'Gentle') return ['gentle', 1];
  if (label.startsWith('Cuff') && !isNaN(Number(label[4]))) {
    const cuffIndex = Number(label[4]);
    return [`cuff${cuffIndex}`, cuffIndex + 1];
  }
  return ['cuff', 0];
}
// 計算 IOU
function iou(box1: number[], box2: number[]): number {
  const [x1, y1, x2, y2] = box1;
  const [x1g, y1g, x2g, y2g] = box2;

  const interX1 = Math.max(x1, x1g);
  const interY1 = Math.max(y1, y1g);
  const interX2 = Math.min(x2, x2g);
  const interY2 = Math.min(y2, y2g);

  const interArea = Math.max(0, interX2 - interX1) * Math.max(0, interY2 - interY1);
  const box1Area = (x2 - x1) * (y2 - y1);
  const box2Area = (x2g - x1g) * (y2g - y1g);
  const unionArea = box1Area + box2Area - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

// 非極大值抑制 (NMS)
function nonMaxSuppressionJS(boxes: number[][], scores: number[], iouThreshold: number = 0.45, maxDetections: number = 100): number[] {
  const idxs = scores
    .map((score, i) => ({ score, i }))
    .sort((a, b) => b.score - a.score)
    .map((item) => item.i);

  const selected: number[] = [];
  while (idxs.length > 0 && selected.length < maxDetections) {
    const current = idxs.shift()!;
    selected.push(current);

    for (let i = idxs.length - 1; i >= 0; i--) {
      const j = idxs[i];
      if (iou(boxes[current], boxes[j]) > iouThreshold) {
        idxs.splice(i, 1);
      }
    }
  }
  return selected;
}
/**
 * 裁切 EN box 並進行 OCR，結果寫入 existingData
 * @param enBoxes 要裁切的 box 陣列 [[x1,y1,x2,y2], ...]
 * @param image SkImage 圖片物件
 * @param imageName 對應 existingData 的 key
 * @param existingData OCR 結果存放物件
 */
export async function cropAndRecognizeENBox(
  enBoxes: number[][],
  image: SkImage,
  imageName: string,
  existingData: {
    [imageName: string]: {
      Text: string[];
    };
  },
) {
  const MIN_OCR_SIZE = 32; // ML Kit 最小尺寸
  if (!enBoxes.length || !image) return;
  // 找最上方的 EN box
  const [x1, y1, x2, y2] = enBoxes.reduce((prev, curr) => (curr[1] < prev[1] ? curr : prev));
  // 計算寬高
  let width = x2 - x1;
  let height = y2 - y1;
  // 如果寬或高小於最小值，增加 padding
  const padX = Math.max(0, MIN_OCR_SIZE - width) / 2;
  const padY = Math.max(0, MIN_OCR_SIZE - height) / 2;

  const cropX1 = Math.max(0, x1 - padX);
  const cropY1 = Math.max(0, y1 - padY);
  const cropX2 = Math.min(image.width(), x2 + padX);
  const cropY2 = Math.min(image.height(), y2 + padY);

  width = cropX2 - cropX1;
  height = cropY2 - cropY1;
  if (width < MIN_OCR_SIZE || height < MIN_OCR_SIZE) {
    console.warn('Box too small for OCR even after padding, skipping:', [x1, y1, x2, y2]);
    return;
  }
  // 裁切圖片
  const croppedSurface = Skia.Surface.MakeOffscreen(width, height);
  if (!croppedSurface) return;
  const canvas = croppedSurface.getCanvas();
  canvas.clear(Skia.Color('white'));
  const paint = Skia.Paint();
  canvas.drawImageRect(image, { x: cropX1, y: cropY1, width, height }, { x: 0, y: 0, width, height }, paint);
  const croppedSnapshot = croppedSurface.makeImageSnapshot();
  const pngBytes = croppedSnapshot.encodeToBytes(); // Uint8Array
  if (!pngBytes || !pngBytes.length) return;
  // 轉 Base64 並寫檔
  const base64String = fromByteArray(pngBytes);
  const filePath = `${DocumentDirectoryPath}/en_crop_${imageName}.png`;
  await writeFile(filePath, base64String, 'base64');
  try {
    const result = await TextRecognition.recognize(`file://${filePath}`);
    const textOnly = result.blocks.map((block) => block.text);
    // 將結果寫入 existingData
    if (!existingData[imageName]) existingData[imageName] = { Text: [] };
    existingData[imageName].Text.push(...textOnly);
  } catch (err) {
    console.error('OCR failed:', err);
  }
}
interface IconData {
  ihb: number;
  cuff: number;
  gentle: number;
}
export interface OutputData {
  icon: IconData;
  text: Record<string, string>;
  number: Record<string, string>;
}

interface BoxResult {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}
/**
 * ONNX 推理 + 輸出整理 (NMS 前)
 */
const Labels = ['text', 'Cuff1', 'Cuff0', 'IHB', 'EN', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Gentle'];
export const predictCustomOnnx = async (
  image: SkImage | null,
  labels: string[] = Labels,
  targetSize: number = 640,
  probThreshold = 0.5,
  iouThreshold: number = 0.45,
): Promise<{
  data: OutputData;
}> => {
  try {
    if (!image) throw new Error('Image not ready');
    //load model
    const best1Path = RNFS.DocumentDirectoryPath + '/best1.onnx';
    await RNFS.copyFileAssets('best1.onnx', best1Path);
    const model = await InferenceSession.create(best1Path);

    if (!model || labels.length === 0) throw new Error('Model or labels not ready');

    // 前處理
    const { tensor, origW, origH } = await preprocessCustomONNX(image, 1, targetSize);
    // 準備 feeds
    const inputName = model.inputNames[0];
    const feeds = { [inputName]: tensor };

    // 執行 ONNX 推理
    const output = await model.run(feeds);
    const outputTensor: Tensor = output[model.outputNames[0]];

    // Tensor → Float32Array
    const outputArray = new Float32Array(outputTensor.data as Float32Array);
    const outputShape = outputTensor.dims; // e.g., [N, M]
    console.log('Raw output tensor dims:', outputTensor.dims, 'first 10 values:');
    // 處理 batch
    const batchOutput: Float32Array = outputShape.length === 2 ? outputArray : outputArray;
    // 處理簡化格式，支援 6 或 7 欄
    const numCols = outputShape[1] || batchOutput.length;
    let standardizedOutput: Float32Array;
    if (numCols === 6 || numCols === 7) {
      standardizedOutput = new Float32Array((batchOutput.length / numCols) * 21); // 21 = 4 bbox + 1 conf + 16~80 classes
      for (let i = 0; i < batchOutput.length / numCols; i++) {
        const base = i * numCols;
        // prettier-ignore
        let x1 = 0, y1 = 0, x2 = 0, y2 = 0,conf = 0,clsIdx = 0;
        if (numCols === 6) {
          // Python 6欄: [x1, y1, x2, y2, conf, class_idx]
          x1 = batchOutput[base + 0];
          y1 = batchOutput[base + 1];
          x2 = batchOutput[base + 2];
          y2 = batchOutput[base + 3];
          conf = batchOutput[base + 4];
          clsIdx = Math.floor(batchOutput[base + 5]);
        } else if (numCols === 7) {
          // 新 7 欄: [0?, x1, y1, x2, y2, class_idx, conf]
          x1 = batchOutput[base + 1];
          y1 = batchOutput[base + 2];
          x2 = batchOutput[base + 3];
          y2 = batchOutput[base + 4];
          clsIdx = Math.floor(batchOutput[base + 5]);
          conf = batchOutput[base + 6];
        }
        // 填入 standardizedOutput
        standardizedOutput[i * 21 + 0] = x1;
        standardizedOutput[i * 21 + 1] = y1;
        standardizedOutput[i * 21 + 2] = x2;
        standardizedOutput[i * 21 + 3] = y2;
        standardizedOutput[i * 21 + 4] = conf;
        if (clsIdx < 80) standardizedOutput[i * 21 + 5 + clsIdx] = 1.0;
      }
    } else if (numCols === 85) {
      standardizedOutput = batchOutput; // 已經是標準格式
    } else {
      console.warn(`Unexpected ONNX output columns: ${numCols}, passing through original data`);
      standardizedOutput = batchOutput;
    }

    // 解析 boxes + scores + class index
    const numBoxes = standardizedOutput.length / 21;
    const boxes: [number, number, number, number][] = [];
    const scores: number[] = [];
    const classIndices: number[] = [];

    for (let i = 0; i < numBoxes; i++) {
      const base = i * 21;
      const x1 = standardizedOutput[base + 0];
      const y1 = standardizedOutput[base + 1];
      const x2 = standardizedOutput[base + 2];
      const y2 = standardizedOutput[base + 3];
      const conf = standardizedOutput[base + 4];
      // 找 class index 最大的
      let maxClsScore = 0;
      let clsIdx = 0;
      for (let c = 0; c < 16; c++) {
        // 可依實際 class 數量調整
        const score = standardizedOutput[base + 5 + c];
        if (score > maxClsScore) {
          maxClsScore = score;
          clsIdx = c;
        }
      }

      const finalScore = conf * maxClsScore;
      if (finalScore < probThreshold) continue;

      boxes.push([x1, y1, x2, y2]);
      scores.push(finalScore);
      classIndices.push(clsIdx);
    }
    console.log(
      'classIndices labels:',
      classIndices.map((a) => labels[a]),
    );
    console.log(
      'scores:',
      scores.map((a) => Math.max(a)),
    );
    // NMS
    const selectedIdxs = nonMaxSuppressionJS(boxes, scores, iouThreshold);

    const scaledBoxes = scaleCoords(
      selectedIdxs.map((i) => boxes[i]),
      [targetSize, targetSize],
      [origH, origW],
    );
    // 組結果
    interface ImageData {
      Text: string[];
      ICON: IconData;
      numbers: string[];
    }
    const existingData: Record<string, ImageData> = {};
    const results: BoxResult[] = [];
    const enBoxes: [number, number, number, number][] = [];
    const imageName = 'image';

    for (let idx = 0; idx < selectedIdxs.length; idx++) {
      const i = selectedIdxs[idx];
      const [x1, y1, x2, y2] = scaledBoxes[idx].map(Math.round);
      const clsIdx = classIndices[i];
      const label = labels[clsIdx] || 'unknown';
      if (!existingData[imageName]) {
        existingData[imageName] = {
          Text: [],
          ICON: { ihb: 0, cuff: 0, gentle: 0 },
          numbers: [],
        };
      }

      if (label === 'text') continue;
      if (label === 'EN') {
        enBoxes.push([x1, y1, x2, y2]);
        continue;
      }
      if (['IHB', 'Cuff0', 'Cuff1', 'Cuff2', 'Cuff3', 'Gentle'].includes(label)) {
        const [key, value] = mapIconLabel(label);
        if (key === 'ihb') existingData[imageName].ICON.ihb = 1;
        else if (key === 'gentle') existingData[imageName].ICON.gentle = 1;
        else existingData[imageName].ICON.cuff = value;
        continue;
      }
      if (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].includes(label)) {
        if (
          !isOverlap(
            [x1, y1, x2, y2],
            results.map((r) => r.bbox),
          )
        ) {
          results.push({ label, confidence: scores[i], bbox: [x1, y1, x2, y2] });
        }
        continue;
      }

      if (
        !isOverlap(
          [x1, y1, x2, y2],
          results.map((r) => r.bbox),
        )
      ) {
        results.push({ label, confidence: scores[i], bbox: [x1, y1, x2, y2] });
      }
    }
    // OCR
    await cropAndRecognizeENBox(enBoxes, image, imageName, existingData);
    // numbers 合併
    const groupedBoxes = groupByYLevel(results);
    const numbers: string[] = [];
    for (const group of groupedBoxes) {
      const sortedGroup = group.sort((a, b) => a.bbox[0] - b.bbox[0]);
      const numberStr = sortedGroup.map((b) => b.label).join('');
      if (!numberStr || numberStr.length >= 4) continue;
      if (numberStr.length >= 2) {
        const n = parseInt(numberStr, 10);
        if (!isNaN(n) && n >= 40) numbers.push(numberStr);
      }
    }
    existingData[imageName].numbers.push(...numbers);
    // 最終輸出
    const outputs: {
      data: OutputData;
    } = {
      data: {
        icon: existingData[imageName].ICON,
        text: {},
        number: {},
      },
    };
    existingData[imageName].Text.forEach((text: string, idx: number) => {
      outputs.data.text[`label${idx + 1}`] = text;
    });

    existingData[imageName].numbers.forEach((num: string, idx: number) => {
      outputs.data.number[`order${idx + 1}`] = num;
    });
    console.log('Final output:', outputs);
    return outputs;
  } catch (err) {
    console.error(err);
    return {
      data: {
        icon: {
          ihb: 0,
          cuff: 0,
          gentle: 0,
        },
        text: {},
        number: {},
      },
    };
  }
};
