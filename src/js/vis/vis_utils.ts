export type LayerType = 'dense' | 'output';

export interface NNLayer {
    id: string;
    type: LayerType;
    neurons: number;
    activation: string;
    element: HTMLElement;
}

export function getLayerColor(type: LayerType): string {
    const colors = {
        // input: '#4CAF50',
        dense: '#2196F3',
        // conv: '#FF9800',
        output: '#F44336'
    };
    return colors[type] || '#999';
}
