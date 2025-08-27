/**
 * 核心推理函式
 */

/**
 * 解析模型輸出
 */
const extractBBFiltered = (
  output: Float32Array | number[],
  outputShape: number[],
  anchors: number[][],
  probThreshold = 0.0001,
) => {
  'worklet';
  const [height, width, channels] = outputShape;
  const numAnchor = anchors.length;
  const numClass = channels / numAnchor - 5;
  const sigmoid = (x: number) => (x > 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
  const arr: number[] = Array.isArray(output) ? output : Array.from(output);
  const results: { box: number[]; classIdx: number; score: number; obj: number }[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let a = 0; a < numAnchor; a++) {
        const base = ((y * width + x) * numAnchor + a) * (numClass + 5);

        const cx = (sigmoid(arr[base + 0]) + x) / width;
        const cy = (sigmoid(arr[base + 1]) + y) / height;

        const w = (Math.exp(arr[base + 2]) * anchors[a][0]) / width;
        const h = (Math.exp(arr[base + 3]) * anchors[a][1]) / height;

        const box = [cx - w / 2, cy - h / 2, w, h];
        const obj = sigmoid(arr[base + 4]);

        const classLogits = arr.slice(base + 5, base + 5 + numClass);
        const maxLogit = Math.max(...classLogits);
        const expScores = classLogits.map((v) => Math.exp(v - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const classProbs = expScores.map((v) => v / sumExp);

        const bestClassProb = Math.max(...classProbs);
        const classIdx = classProbs.indexOf(bestClassProb);
        const score = obj * bestClassProb;
        const result = { box, classIdx, score, obj };

        if (score > probThreshold) results.push(result); // <-- 存 obj
      }
    }
  }
  console.log(results);
  return results;
};
/**
 * NMS
 */
const nonMaxSuppression = (boxes: number[][], scores: number[], iouThreshold = 0.45, maxDetections = 20) => {
  'worklet';
  const selectedBoxes: number[][] = [];
  const selectedIndices: number[] = [];
  const selectedScores: number[] = [];

  const areas = boxes.map((b) => b[2] * b[3]);

  let idxs = scores
    .map((_, i) => i)
    .filter((i) => scores[i] > 0)
    .sort((a, b) => scores[b] - scores[a]);

  while (selectedBoxes.length < maxDetections && idxs.length > 0) {
    const i = idxs.shift()!;
    selectedBoxes.push(boxes[i]);
    selectedIndices.push(i);
    selectedScores.push(scores[i]);

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

  return { selectedBoxes, selectedIndices, selectedProbs: selectedScores };
};
export const predictImageRN = (
  model: any,
  labels: string[],
  inputImage: Float32Array,
  outputShape: number[],
  probThreshold = 0.1,
  maxDetections = 20,
) => {
  'worklet';

  const output: any = model.runSync([inputImage])[0]; //[[HxWxC]]
  const ANCHORS = [
    [0.573, 0.677],
    [1.87, 2.06],
    [3.34, 5.47],
    [7.88, 3.53],
    [9.77, 9.17],
  ];

  const extractResults = extractBBFiltered(output, outputShape, ANCHORS, probThreshold);

  if (extractResults.length === 0) return [];
  // 分組取最高分
  const bestByClass: Record<number, (typeof extractResults)[0]> = {};
  for (const r of extractResults) {
    const score = r.score;
    if (!bestByClass[r.classIdx] || score > bestByClass[r.classIdx].score) {
      bestByClass[r.classIdx] = r;
    }
  }
  // 轉回陣列
  const filteredResults = Object.values(bestByClass);

  // 準備 NMS
  const boxes = filteredResults.map((r) => r.box);
  const scores = filteredResults.map((r) => r.score);
  const classes = filteredResults.map((r) => r.classIdx);
  const { selectedBoxes, selectedIndices, selectedProbs } = nonMaxSuppression(boxes, scores, 0.45, 100);
  const selectedClasses = selectedIndices.map((i) => classes[i]);

  return selectedBoxes.map((box, i) => ({
    label: labels[selectedClasses[i]],
    prob: selectedProbs[i],
    box,
  }));
};
