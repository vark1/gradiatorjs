import { LayerType } from "../types_and_interfaces/general.js";

export function getLayerColor(type: LayerType): string {
    const colors = {
        dense: '#2196F3',
        conv: '#FF9800',
        flatten: '#8cf800',
        maxpool: '#b5d3f2'
    };
    return colors[type] || '#999';
}