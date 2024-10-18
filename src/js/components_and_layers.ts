// import { activationfn } from "./activations";
// import { add, dot } from "./Val/ops";
// import { Tensor } from "./nd_old/tensor";
// import {ndrandom} from "./nd_old/utils_nd";

// export class Neuron {
//     weights: Tensor
//     bias: Tensor
//     nonlin: boolean

//     constructor (nin: Tensor, nonlin: boolean = true){
//         this.weights = new Tensor(ndrandom(nin.shape), 'w') // TODO: Make this 'normal dis'
//         this.nonlin = nonlin
//         this.bias = new Tensor(0, 'bias')
//     }

//     public forward(input: Tensor) {
//         let z = add(dot(this.weights, input), this.bias)
//         return this.nonlin ? activationfn(z) : z;
//     }
// }

// // FC
// export class FCLayer {
//     neurons: Neuron[]

//     constructor (num_neurons: number) {
//         this.neurons = new Array<Neuron>(num_neurons)
//     }
// }

// // Softmax 

// // Pooling layers: MAX, AVG

// // Convolutional layer

// // Neurons