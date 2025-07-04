import { Val } from "../Val/val.js";

export type LayerType = 'dense' | 'conv' | 'flatten' | 'maxpool';
export type ActivationType = 'relu' | 'sigmoid' | 'tanh' | 'softmax';
export type NNLayer = DenseNNLayer | ConvNNLayer | FlattenNNLayer | MaxPool2DLayer;

interface BaseNNLayer {
    id: string;
    type: LayerType;
    element: HTMLElement;
}
interface DenseNNLayer extends BaseNNLayer {
    type: 'dense';
    neurons: number;
    activation: ActivationType;
}
interface ConvNNLayer extends BaseNNLayer {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: ActivationType;
}
interface FlattenNNLayer extends BaseNNLayer {
    type: 'flatten';
}
interface MaxPool2DLayer extends BaseNNLayer {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}

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