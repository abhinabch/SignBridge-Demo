declare module '@tensorflow/tfjs-tflite' {
    export function loadTFLiteModel(url: string | ArrayBuffer): Promise<TFLiteModel>;
    export interface TFLiteModel {
        predict(inputs: any, config?: any): any;
    }
}
