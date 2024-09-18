import { activationfn } from "./activations";
import { add, dot } from "./ops";
import { Tensor } from "./tensor";

export class Neuron {
    weights: Tensor
    bias: Tensor
    nonlin: boolean

    constructor (nin: Tensor, nonlin: boolean = true){
        this.weights = Tensor.random(nin.shape) // TODO: Make this uniform dis
        this.nonlin = nonlin
        this.bias = new Tensor(0, 'bias')
    }

    public forward(inputs: Tensor) {
        let z = add(dot(this.weights, inputs), this.bias)
        return this.nonlin ? activationfn(z) : z;
    }
}

// FC

// Softmax 

// Pooling layers: MAX, AVG

// Convolutional layer

// Neurons