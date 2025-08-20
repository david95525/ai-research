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
export const nonMaxSuppression = (boxes: number[][], scores: number[], iouThreshold = 0.45, maxDetections = 20) => {
  const selected: number[] = [];
  const areas = boxes.map((b) => b[2] * b[3]);

  const idxs = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);

  while (selected.length < maxDetections && idxs.length > 0) {
    const i = idxs.shift()!;
    if (scores[i] < 0.1) break;
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
export function dumpTensor(
  result: any,
  opts: { name?: string; preview?: number } = {}
): string {
  const { name = "Tensor", preview = 10 } = opts;

  const getShape = (arr: any): number[] => {
    if (!Array.isArray(arr) && !(arr instanceof Float32Array || arr instanceof Int32Array)) {
      return [];
    }
    let shape: number[] = [];
    let current: any = arr;
    while (Array.isArray(current) || current?.length !== undefined) {
      shape.push(current.length);
      current = current[0];
    }
    return shape;
  };

  let shape = getShape(result);
  let flat: number[] = [];

  try {
    if (Array.isArray(result)) {
      flat = result.flat(Infinity) as number[];
    } else if (result instanceof Float32Array || result instanceof Int32Array || result instanceof Uint8Array) {
      flat = Array.from(result);
    } else {
      flat = [result];
    }
  } catch {
    flat = [];
  }

  return [
    `=== ${name} Dump ===`,
    `Type: ${Object.prototype.toString.call(result)}`,
    `Shape: [${shape.join(", ")}]`,
    `Length: ${flat.length}`,
    `Values (first ${Math.min(preview, flat.length)}): ${flat.slice(0, preview).join(", ")}`
  ].join("\n");
}