/**
 * 驗證 ONNX 前處理結果
 * - 檢查像素範圍 [0,1]
 * - 檢查 resize 與 padding
 * - 隨機抽樣部分 grid cell RGB
 */
export function validatePreprocess(
  floatData: Float32Array,
  targetSize: number,
  origW: number,
  origH: number,
  padLeft: number,
  padTop: number,
  sampleCells: number[] = [0, Math.floor(targetSize / 2), targetSize - 1],
) {
  console.log('--- Preprocess Validation ---');
  console.log(`Original size: ${origW}x${origH}`);
  console.log(`Target size: ${targetSize}x${targetSize}`);
  console.log(`Padding: left=${padLeft}, top=${padTop}`);
  const scale = Math.min(targetSize / origW, targetSize / origH);
  console.log(`Scale: ${scale.toFixed(4)}`);

  // 檢查 pixel 範圍
  let minVal = 1,
    maxVal = 0;
  for (let i = 0; i < floatData.length; i++) {
    if (floatData[i] < minVal) minVal = floatData[i];
    if (floatData[i] > maxVal) maxVal = floatData[i];
  }
  console.log(`Float32 RGB range: min=${minVal.toFixed(4)}, max=${maxVal.toFixed(4)}`);

  // 抽樣檢查 grid cell RGB
  console.log('Sample RGB values at selected rows/cols:');
  for (const y of sampleCells) {
    for (const x of sampleCells) {
      const idx = (y * targetSize + x) * 3;
      const r = floatData[idx].toFixed(3);
      const g = floatData[idx + 1].toFixed(3);
      const b = floatData[idx + 2].toFixed(3);
      console.log(`  Pixel (x=${x}, y=${y}): R=${r}, G=${g}, B=${b}`);
    }
  }

  // Optional: visualize padding areas
  const padRight = targetSize - padLeft - Math.round(origW * scale);
  const padBottom = targetSize - padTop - Math.round(origH * scale);

  console.log('Expected non-zero area (after padding):');
  console.log(`  X range: ${padLeft} ~ ${padLeft + Math.round(origW * scale) - 1}`);
  console.log(`  Y range: ${padTop} ~ ${padTop + Math.round(origH * scale) - 1}`);
  console.log(`  Padding right: ${padRight}, bottom: ${padBottom}`);

  // Optional: check padding pixels (should be 0)
  const checkPad = (x: number, y: number) => {
    const idx = (y * targetSize + x) * 3;
    return floatData[idx] === 0 && floatData[idx + 1] === 0 && floatData[idx + 2] === 0;
  };
  console.log('Padding check (corner pixels should be 0):');
  console.log(`  Top-left: ${checkPad(0, 0)}, Top-right: ${checkPad(targetSize - 1, 0)}`);
  console.log(
    `  Bottom-left: ${checkPad(0, targetSize - 1)}, Bottom-right: ${checkPad(targetSize - 1, targetSize - 1)}`,
  );
}

/**
 * 驗證 YOLO output 排列方式
 * @param output Float32Array | number[]
 * @param outputShape [H, W, C]
 * @param numAnchor number of anchors
 * @param numSamples number of grid cells to sample (default 3)
 */
export function validateYoloOutput(
  output: Float32Array | number[],
  outputShape: [number, number, number],
  numAnchor: number,
  numSamples = 3,
) {
  const [H, W, C] = outputShape;
  const arr: number[] = Array.isArray(output) ? output : Array.from(output);
  const stride = C / numAnchor;

  console.log('--- Validating YOLO output ---');

  // 隨機挑幾個 grid cell
  const sampleCoords: [number, number][] = [
    [0, 0],
    [Math.floor(H / 2), Math.floor(W / 2)],
    [H - 1, W - 1],
  ];

  for (const [y, x] of sampleCoords) {
    console.log(`Grid cell (y=${y}, x=${x})`);

    for (let a = 0; a < numAnchor; a++) {
      // Grouped by feature
      const tx_feat = arr[((y * W + x) * numAnchor + a) * stride + 0];
      const ty_feat = arr[((y * W + x) * numAnchor + a) * stride + 1];
      const tw_feat = arr[((y * W + x) * numAnchor + a) * stride + 2];
      const th_feat = arr[((y * W + x) * numAnchor + a) * stride + 3];
      const obj_feat = arr[((y * W + x) * numAnchor + a) * stride + 4];

      // Grouped by anchor
      const tx_anchor = arr[((y * W + x) * stride + 0) * numAnchor + a];
      const ty_anchor = arr[((y * W + x) * stride + 1) * numAnchor + a];
      const tw_anchor = arr[((y * W + x) * stride + 2) * numAnchor + a];
      const th_anchor = arr[((y * W + x) * stride + 3) * numAnchor + a];
      const obj_anchor = arr[((y * W + x) * stride + 4) * numAnchor + a];

      console.log(
        `  Anchor ${a}: featureGrouped tx=${tx_feat.toFixed(3)}, ty=${ty_feat.toFixed(3)}, tw=${tw_feat.toFixed(
          3,
        )}, th=${th_feat.toFixed(3)}, obj=${obj_feat.toExponential(3)}`,
      );
      console.log(
        `            anchorGrouped tx=${tx_anchor.toFixed(3)}, ty=${ty_anchor.toFixed(3)}, tw=${tw_anchor.toFixed(
          3,
        )}, th=${th_anchor.toFixed(3)}, obj=${obj_anchor.toExponential(3)}`,
      );
    }
  }

  console.log('提示：觀察哪一種排列 tx/ty/tw/th/obj 的值不全為 0 或極小，這就是正確的索引公式。');
}
