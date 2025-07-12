import { Val } from "./val.js";

export type LayerType = 'dense' | 'conv' | 'flatten' | 'maxpool';
export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'softmax';

export interface MinMaxInfo {
    minv: number;
    maxv: number;
    dv: number; // Range (maxv - minv)
}

export interface NetworkParams {
    loss_fn: (Y_pred: Val, Y_true: Val) => Val,
    l_rate: number,
    epochs: number,
    batch_size: number,
    multiClass: boolean
}

export interface TrainingProgress{
    epoch: number,
    batch_idx: number,
    loss: number,
    accuracy: number, 
    iterTime: number,
    visData: {
        sampleX: Val;
        sampleY_label: number;
        layerOutputs: { Z: Val | null; A: Val | null; }[];
    }
}