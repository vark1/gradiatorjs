import { Val } from './Val/val.js';

export function relu (Z: Val) : Val{
    let out = new Val(Z.shape)
    out.data = Z.data.map((k:number)=> Math.max(k, 0));

    out._prev = new Set([Z]);
    out._backward = () => {
        for(let i=0; i<Z.size; ++i) {
            Z.grad[i] += (Z.data[i] > 0 ? out.grad[i] : 0);
        }
    }
    return out
}

export function sigmoid (Z: Val) : Val {
    let out = new Val(Z.shape)
    out.data = Z.data.map((k:number)=>(1 / (1 + Math.exp(-k))));

    out._prev = new Set([Z]);
    out._backward = () => {
        for (let i=0; i<Z.size; ++i) {
            Z.grad[i] += out.data[i] * (1-out.data[i]) * out.grad[i];
        }
    }
    return out;
}

export function tanh (Z: Val) : Val{
    let out = new Val(Z.shape);
    out.data = Z.data.map((k:number)=>(Math.tanh(k)));

    out._prev = new Set([Z]);
    out._backward = () => {
        for (let i=0; i<Z.size; ++i) {
            Z.grad[i] += (1 - out.data[i] * out.data[i]) * out.grad[i];
        }
    }
    return out
}