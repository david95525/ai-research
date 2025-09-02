import * as RNFS from 'react-native-fs';
import { Skia } from '@shopify/react-native-skia';
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
