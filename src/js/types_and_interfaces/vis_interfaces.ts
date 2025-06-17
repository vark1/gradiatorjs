import { Val } from "Val/val";
import { ActivationType, NNLayer, LayerType } from "./general";

export type LayerCreationOptions = DenseLayerOptions | ConvLayerOptions | MaxPoolLayerOptions | FlattenLayerOptions;

// When loading a network from the localstorage, this defines what parts of the layer data we want to save.
// We explicitly omit the 'element' property as it's not serializable.
export type SerializableNNLayer = Omit<NNLayer, 'element'>;

// options stored in localstorage
interface DenseLayerOptions {
    type: 'dense';
    neurons: number;
    activation: ActivationType;
}
interface ConvLayerOptions {
    type: 'conv';
    out_channels: number;
    kernel_size: number;
    stride: number;
    padding: number;
    activation: ActivationType;
}
interface MaxPoolLayerOptions {
    type: 'maxpool';
    pool_size: number;
    stride: number;
}
interface FlattenLayerOptions {
    type: 'flatten';
}

export interface LayerOutputData {
    Z: Val | null;
    A: Val | null;
}

export interface visPackage {
    sampleX: Val;
    sampleY_label: number;
    layerOutputs: { Z: Val | null; A: Val | null; }[];
}