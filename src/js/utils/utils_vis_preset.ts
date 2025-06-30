import { LayerCreationOptions } from "../types_and_interfaces/vis_interfaces.js";

export const PRESETS: {
    mnist: {network: LayerCreationOptions[], params: {l_rate: number, epochs: number, batch_size: number}}, 
    catvnoncat: {network: LayerCreationOptions[], params: {l_rate: number, epochs: number, batch_size: number}}, 
} = {
    mnist: {
        network: [
            { type: 'conv', out_channels: 8, kernel_size: 5, stride: 1, padding: 2, activation: 'relu' },
            { type: 'maxpool', pool_size: 2, stride: 2 },
            { type: 'conv', out_channels: 16, kernel_size: 5, stride: 1, padding: 2, activation: 'relu' },
            { type: 'maxpool', pool_size: 3, stride: 3 },
            { type: 'flatten' },
            { type: 'dense', neurons: 10, activation: 'softmax' }
        ],
        params: { l_rate: 0.01, epochs: 500, batch_size: 20 }
    },
    catvnoncat: {
        network: [
            { type: 'conv', out_channels: 16, kernel_size: 3, stride: 1, padding: 1, activation: 'relu' },
            { type: 'maxpool', pool_size: 2, stride: 2 },
            { type: 'conv', out_channels: 32, kernel_size: 3, stride: 1, padding: 1, activation: 'relu' },
            { type: 'maxpool', pool_size: 2, stride: 2 },
            { type: 'flatten' },
            { type: 'dense', neurons: 64, activation: 'relu' },
            { type: 'dense', neurons: 1, activation: 'sigmoid' }
        ],
        params: { l_rate: 0.001, epochs: 500, batch_size: 209 }
    }
};