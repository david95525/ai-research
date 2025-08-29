import * as RNFS from 'react-native-fs';
interface scale {
  width: number;
  height: number;
}
import { Skia, ColorType, AlphaType, ImageInfo, SkImage } from '@shopify/react-native-skia';

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
/**
 * - 保持原始比例
 * - 寬高對齊 32 的倍數
 * - 輸出 scale
 */
export const caculateScale = (frame: any, defaultInputSize: number = 416 * 416): scale => {
  'worklet';
  const ratio = Math.sqrt(defaultInputSize / (frame.width * frame.height));
  let newWidth = Math.ceil(frame.width * ratio);
  let newHeight = Math.ceil(frame.height * ratio);

  // 對齊 32 倍數
  newWidth = 32 * Math.ceil(newWidth / 32);
  newHeight = 32 * Math.ceil(newHeight / 32);
  return { width: newWidth, height: newHeight };
};

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
function calcResize32(origW: number, origH: number, target: number = 416) {
  const scale = Math.sqrt((target * target) / (origW * origH));
  let newW = Math.ceil(origW * scale);
  let newH = Math.ceil(origH * scale);

  // 對齊到 32
  newW = 32 * Math.ceil(newW / 32);
  newH = 32 * Math.ceil(newH / 32);

  return { newWidth: newW, newHeight: newH };
}
/**
 * 1. 讀取本地 PNG 並做前處理（縮放 + 對齊 32 的倍數）
 * 回傳 Float32Array，值 normalized 到 [0,1]
 */
export async function preprocessLocalImage(image: SkImage | null): Promise<{
  data: Float32Array;
  width: number;
  height: number;
}> {
  try {
    if (!image) throw new Error('Failed to load image');

    const { newWidth, newHeight } = calcResize32(image.width(), image.height());
    //建立一個空的 surface
    const surface = Skia.Surface.MakeOffscreen(newWidth, newHeight);
    if (!surface) throw new Error('Failed to create surface');
    const canvas = surface.getCanvas();
    const paint = Skia.Paint();
    canvas.drawImageRect(
      image,
      { x: 0, y: 0, width: image.width(), height: image.height() },
      { x: 0, y: 0, width: newWidth, height: newHeight },
      paint,
    );

    //拿到像素 (RGBA Uint8Array)
    const snapshot = surface.makeImageSnapshot();
    // 設定 ImageInfo，直接讀 Float32Array
    const imageInfo: ImageInfo = {
      width: newWidth,
      height: newHeight,
      colorType: ColorType.RGBA_8888, // RGBA
      alphaType: AlphaType.Unpremul, // 非預乘 alpha
    };
    const pixels = snapshot.readPixels(0, 0, imageInfo);
    if (!pixels) throw new Error('Failed to read pixels');
    let floatData: Float32Array;
    if (pixels instanceof Uint8Array) {
      // fallback: Uint8 → Float32
      floatData = new Float32Array(newWidth * newHeight * 3);
      let ptr = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        floatData[ptr++] = pixels[i] / 255;
        floatData[ptr++] = pixels[i + 1] / 255;
        floatData[ptr++] = pixels[i + 2] / 255;
      }
    } else {
      // 如果 already Float32Array，直接取 RGB
      floatData = new Float32Array(newWidth * newHeight * 3);
      let ptr = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        floatData[ptr++] = pixels[i]; // R
        floatData[ptr++] = pixels[i + 1]; // G
        floatData[ptr++] = pixels[i + 2]; // B
      }
    }

    return { data: floatData, width: newWidth, height: newHeight };
  } catch (err) {
    console.error('preprocessLocalImage failed:', err);
    return { data: new Float32Array(0), width: 0, height: 0 };
  }
}
