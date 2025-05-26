export type LayerType = 'dense' | 'conv' |'output';

export interface BaseNNLayer {
    id: string;
    type: LayerType;
    activation: string;
    element: HTMLElement;
}

export interface DenseNNLayer extends BaseNNLayer {
    type: 'dense' | 'output';
    neurons: number;
}

export interface ConvNNLayer extends BaseNNLayer {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
}

export type NNLayer = DenseNNLayer | ConvNNLayer;

export function getLayerColor(type: LayerType): string {
    const colors = {
        dense: '#2196F3',
        conv: '#FF9800',
        output: '#F44336'
    };
    return colors[type] || '#999';
}
