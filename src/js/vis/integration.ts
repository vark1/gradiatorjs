import { Sequential, Dense, Module } from '../nn/nn.js';
import * as afn from '../Val/activations.js';
import { NeuralNetworkVisualizer } from './vis.js';
import { NNLayer } from '../utils/utils_vis.js';

const AFN_MAP = {
    "relu": afn.relu,
    "sigmoid": afn.sigmoid,
    "tanh": afn.tanh
}

export function createEngineModelFromVisualizer(
    visualizer: NeuralNetworkVisualizer,
    inputFeatureSize: number,
): Sequential {

    const errors = visualizer.validateNetwork();
    if (errors.length > 0) {
        throw new Error(`Invalid network configuration: ${errors.join(', ')}`);
    }

    const vizLayers: NNLayer[] = (visualizer as any).layers;

    if (!vizLayers || vizLayers.length === 0) {
        throw new Error("Visualizer has no layers defined.");
    }

    const engineLayers: Module[] = [];
    let currentInputSize = inputFeatureSize;

    for (const vizLayer of vizLayers) {
        const activationFn = AFN_MAP[vizLayer.activation as keyof typeof AFN_MAP];
        let engineLayer: Module;

        switch (vizLayer.type) {
            case 'dense':
                engineLayer = new Dense(
                    currentInputSize,
                    vizLayer.neurons,
                    activationFn
                );
                engineLayers.push(engineLayer);
                currentInputSize = vizLayer.neurons; // Output size of this layer is input size for next
                break;

            case 'output':      // this is also a dense layer, just the last one (likely with 1-10 neurons only)
                engineLayer = new Dense(
                    currentInputSize,
                    vizLayer.neurons,
                    activationFn
                );
                engineLayers.push(engineLayer);
                currentInputSize = vizLayer.neurons;
                break;

            default:
                throw new Error(`Unsupported visualizer layer type: ${vizLayer.type}`);
        }
    }

    if (engineLayers.length === 0) {
        throw new Error("No computational engine layers could be created from the visualizer configuration.");
    }

    const model = new Sequential(...engineLayers);
    return model;
}