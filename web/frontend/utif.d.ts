declare module 'utif' {
  export interface TiffPage {
    width: number;
    height: number;
    data?: Uint8Array;
    subIFD?: TiffPage[];
    t256?: number[];
    t257?: number[];
    t258?: number[];
    [key: string]: unknown;
  }

  export function decode(buff: ArrayBuffer): TiffPage[];
  export function decodeImage(buff: ArrayBuffer, img: TiffPage, ifds: TiffPage[]): void;
  export function toRGBA8(out: TiffPage): Uint8Array;
}
