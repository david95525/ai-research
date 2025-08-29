/**
 * 核心推理函式
 */

/**
 * 解析模型輸出
 */
const extractBBFiltered = (
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
        // layout B: grouped-by-feature
        const tx = arr[((y * width + x) * stride + 0) * numAnchor + a];
        const ty = arr[((y * width + x) * stride + 1) * numAnchor + a];
        const tw = arr[((y * width + x) * stride + 2) * numAnchor + a];
        const th = arr[((y * width + x) * stride + 3) * numAnchor + a];
        const tobj = arr[((y * width + x) * stride + 4) * numAnchor + a];

        const cx = (sigmoid(tx) + x) / width;
        const cy = (sigmoid(ty) + y) / height;
        const w = (Math.exp(tw) * anchors[a][0]) / width;
        const h = (Math.exp(th) * anchors[a][1]) / height;
        const box = [cx - w / 2, cy - h / 2, w, h];

        const obj = sigmoid(tobj);

        // class logits
        const classLogits: number[] = [];
        for (let k = 0; k < numClass; k++) {
          classLogits.push(arr[((y * width + x) * stride + 5 + k) * numAnchor + a]);
        }

        // softmax
        const maxLogit = Math.max(...classLogits);
        const expScores = classLogits.map((v) => Math.exp(v - maxLogit));
        const sumExp = expScores.reduce((a, b) => a + b, 0);
        const classProbs = expScores.map((v) => (v / sumExp) * obj);

        // 判斷是否保留（用最大 class prob）
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

export const predictImageRN = (
  model: any,
  labels: string[],
  inputImage: Float32Array,
  outputShape: number[],
  probThreshold = 0.5,
  iouThreshold = 0.3,
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

  const boxes = extractResults.map((r) => r.box);
  const classProbs = extractResults.map((r) => r.score);

  const { selectedBoxes, selectedClasses, selectedProbs } = nonMaxSuppression(
    boxes,
    classProbs,
    probThreshold,
    iouThreshold,
    maxDetections,
  );
  console.log(selectedClasses);
  return selectedBoxes.map((box, i) => ({
    label: labels[selectedClasses[i]],
    prob: selectedProbs[i],
    box,
  }));
};
