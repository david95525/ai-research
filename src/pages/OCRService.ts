import TextRecognition from '@react-native-ml-kit/text-recognition';

export async function runOCR(frame: any): Promise<string> {
  try {
    const result = await TextRecognition.recognize(frame);
    return result.blocks.map(block => block.text).join('\n');
  } catch (err) {
    console.error('OCR failed:', err);
    throw err;
  }
}
