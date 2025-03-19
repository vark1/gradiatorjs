export type LayerType = 'input' | 'dense' | 'conv' | 'output';

export interface NNLayer {
    id: string;
    type: LayerType;
    neurons: number;
    activation: string;
    element: HTMLElement;
}

export interface NetworkConfig {
    layer_sizes: number[];
    activations: string[];
}

export function getLayerColor(type: LayerType): string {
    const colors = {
        input: '#4CAF50',
        dense: '#2196F3',
        conv: '#FF9800',
        output: '#F44336'
    };
    return colors[type] || '#999';
}
