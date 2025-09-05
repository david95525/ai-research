export interface IconData {
  ihb: number;
  cuff: number;
  gentle: number;
}

export interface OutputData {
  icon: IconData;
  text: Record<string, string>;
  number: Record<string, string>;
}

export interface BoxResult {
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
}