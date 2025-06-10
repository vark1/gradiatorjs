export type LayerType = 'dense' | 'conv' | 'flatten' | 'maxpool';
export type NNLayer = DenseNNLayer | ConvNNLayer | FlattenNNLayer | MaxPool2DLayer;

export interface BaseNNLayer {
    id: string;
    type: LayerType;
    element: HTMLElement;
}

export interface DenseNNLayer extends BaseNNLayer {
    type: 'dense';
    neurons: number;
    activation: string;
}

export interface ConvNNLayer extends BaseNNLayer {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: string;
}

export interface FlattenNNLayer extends BaseNNLayer {
    type: 'flatten';
}

export interface MaxPool2DLayer extends BaseNNLayer {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}

export interface VISActivationData {
    layerIdx: number;
    layerType: LayerType;
    shape: number[];
    activationSample: Float64Array;
}

export interface MinMaxInfo {
    minv: number;
    maxv: number;
    dv: number; // Range (maxv - minv)
}