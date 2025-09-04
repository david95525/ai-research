import * as RNFS from 'react-native-fs';
import { Skia, SkCanvas } from '@shopify/react-native-skia';
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
export const loadImage = async () => {
  try {
    const base64 = await RNFS.readFileAssets('scanbp.jpg', 'base64');
    const data = Skia.Data.fromBase64(base64);
    const image = Skia.Image.MakeImageFromEncoded(data);
    if (!image) throw new Error('Failed to load image');
    return image;
  } catch (err) {
    console.error('copyAssetToDocDir failed:', err);
    throw new Error('Failed to load image');
  }
};
export const calcResize32 = (origW: number, origH: number, target: number = 416) => {
  const scale = Math.sqrt((target * target) / (origW * origH));
  let newW = Math.ceil(origW * scale);
  let newH = Math.ceil(origH * scale);
  // 對齊到 32
  newW = 32 * Math.ceil(newW / 32);
  newH = 32 * Math.ceil(newH / 32);
  return { newWidth: newW, newHeight: newH };
};
export function applyOrientation(canvas: SkCanvas, orientation: number, width: number, height: number) {
  const cx = width / 2;
  const cy = height / 2;
  switch (orientation) {
    case 2: // 水平翻轉
      canvas.translate(width, 0);
      canvas.scale(-1, 1);
      break;
    case 3: // 旋轉 180
      canvas.rotate(180, cx, cy);
      break;
    case 4: // 垂直翻轉
      canvas.translate(0, height);
      canvas.scale(1, -1);
      break;
    case 5: // 垂直翻轉 + 旋轉 90 CW
      canvas.rotate(90, cx, cy);
      canvas.translate(0, height);
      canvas.scale(1, -1);
      break;
    case 6: // 旋轉 90 CW
      canvas.rotate(90, cx, cy);
      break;
    case 7: // 水平翻轉 + 旋轉 90 CW
      canvas.rotate(90, cx, cy);
      canvas.scale(-1, 1);
      canvas.translate(width, 0);
      break;
    case 8: // 旋轉 270 CW
      canvas.rotate(270, cx, cy);
      break;
    default: // orientation=1 → 不動
      break;
  }
}
/**
 * HWC -> CHW 轉換
 */
export function hwcToChw(data: Float32Array, width: number, height: number): Float32Array {
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
