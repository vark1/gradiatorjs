import { Tensor } from './tensor';
import { NDArray, t_any } from './types';
import { applyFnUnary, convertToTensor } from './utils';
import {add, sub, pow, mul} from './ops'

const ACT_MAP = {
    "relu": (x: number) => Math.max(x, 0),
    "sigmoid": (x: number) => 1 / (1 + Math.exp(-x)),
    "tanh": (x: number) => (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1)
};

const DER_MAP = {
    "relu" : (x: NDArray) => applyFnUnary(x, (v) => (v === 0 ? 0 : 1)),
    "sigmoid": (x: NDArray) => mul(x, sub(1, x)),
    "tanh": (x: NDArray) => sub(1, pow(x, 2))
}


// Main function to apply activation function
export function activationfn(t: t_any, type: keyof typeof ACT_MAP = 'relu'): Tensor {
    let t_ = convertToTensor(t);
    const data = applyFnUnary(t_.data, ACT_MAP[type]);
    
    let out = new Tensor(data, `${type}(${t_.label})`, t_.shape, type, [t_]);

    // Backward pass using derivative
    out._backward = () => {
        let derivative = mul(DER_MAP[type](data), out.grad);
        t_.grad = add(t_.grad, derivative).data;
    };

    return out;
}