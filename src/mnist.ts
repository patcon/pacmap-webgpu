/**
 * MNIST loader. Uses the TF.js-hosted sprite: a 784 x 65000 PNG where each row
 * is one digit flattened to 784 grayscale pixels. Avoids having to gunzip and
 * parse the idx format in the browser.
 *
 * [If this bucket ever goes away, drop a copy of the PNG + label file in
 *  /public and point SPRITE_URL / LABELS_URL at the local paths.]
 */

const SPRITE_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const LABELS_URL =
  "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

export const IMAGE_SIZE = 784;
export const NUM_AVAILABLE = 65000;

export interface Mnist {
  /** n x 784, values in [0, 1] */
  X: Float32Array;
  /** n digit labels, 0-9 */
  labels: Uint8Array;
  n: number;
}

export async function loadMnist(
  n: number,
  onStatus: (msg: string) => void = () => {}
): Promise<Mnist> {
  if (n > NUM_AVAILABLE) throw new Error(`max ${NUM_AVAILABLE} samples`);

  onStatus("Fetching MNIST sprite…");

  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("failed to load MNIST sprite"));
    img.src = SPRITE_URL;
  });

  onStatus("Decoding pixels…");

  // The full sprite decodes to ~200MB of RGBA, so pull it out in row chunks
  // rather than allocating one canvas the height of the whole image.
  const CHUNK = 2500;
  const canvas = document.createElement("canvas");
  canvas.width = IMAGE_SIZE;
  canvas.height = Math.min(CHUNK, n);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("no 2d context");

  const X = new Float32Array(n * IMAGE_SIZE);
  let done = 0;
  while (done < n) {
    const rows = Math.min(CHUNK, n - done);
    ctx.clearRect(0, 0, IMAGE_SIZE, canvas.height);
    ctx.drawImage(img, 0, done, IMAGE_SIZE, rows, 0, 0, IMAGE_SIZE, rows);
    const px = ctx.getImageData(0, 0, IMAGE_SIZE, rows).data;
    const base = done * IMAGE_SIZE;
    for (let i = 0; i < rows * IMAGE_SIZE; i++) {
      X[base + i] = px[i * 4] / 255; // grayscale: R == G == B
    }
    done += rows;
    onStatus(`Decoding pixels… ${done}/${n}`);
  }

  onStatus("Fetching labels…");

  // Labels are stored one-hot, 10 uint8 per example.
  const raw = new Uint8Array(await (await fetch(LABELS_URL)).arrayBuffer());
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 10; k++) {
      if (raw[i * 10 + k]) {
        labels[i] = k;
        break;
      }
    }
  }

  return { X, labels, n };
}
