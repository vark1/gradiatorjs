import { Val } from './Val/val.js';
import * as ops from './Val/ops.js'

const DER_MAP = {
    "relu" : (x: number) => (x === 0 ? 0 : 1),
    "sigmoid": (x: number) => x * (1-x),
    "tanh": (x: number) => 1 - x**2
}

export function relu (Z: Val) {
    let x = new Val(Z.shape)
    x.data = Z.data.map((k:number)=> Math.max(k, 0))
    return x
}

export function sigmoid (Z: Val) {
    let x = new Val(Z.shape)
    x.data = Z.data.map((k:number)=>(1 / (1 + Math.exp(-k))))
    return x
}

export function tanh (Z: Val) {
    let x = new Val(Z.shape)
    x.data = Z.data.map((k:number)=>(Math.tanh(k)))
    return x
}

// backward propagation for as single RELU unit
export function reluBackward (dA: Val, cache: Val) : Val {
    let Z = cache
    let dZ = dA.clone()
    for(let i=0; i<Z.size; i++) {
        if (Z.data[i]<=0) {
            dZ.data[i] = 0
        }
    }
    return dZ

    // let x = new Val(dA.shape)
    // x.data = dA.data.map((k: number) => ((k === 0 ? 0 : 1)))
    // return [x, dA]
}

// backward propagation for a single sigmoid unit
export function sigmoidBackward(dA: Val, cache: Val) : Val {
    let Z = cache
    let s = ops.pow(ops.add(1, ops.exp(ops.negate(Z))), -1)
    let dZ = ops.mul(ops.mul(dA, s), ops.sub(1, s))
    return dZ
    // let x = new Val(dA.shape)
    // x.data = dA.data.map((k:number)=> (k * (1-k)))
    // return [x, dA]
}

// const ACT_MAP = {
//     "relu": (x: number) => Math.max(x, 0),
//     "sigmoid": (x: number) => 1 / (1 + Math.exp(-x)),
//     "tanh": (x: number) => Math.tanh(x)
// };

// export function activationfn(t: Val, type: keyof typeof ACT_MAP = 'relu'): Val {
//     let x = new Val(t.shape)
//     x.data = t.data.map((k: number) => ACT_MAP[type](k))
    
//     // Backward pass using derivative
//     // out._backward = () => {
//     //     let derivative = mul(DER_MAP[type](data), out.grad);
//     //     t_.grad = add(t_.grad, derivative).data;
//     // };

//     return x;
// }
