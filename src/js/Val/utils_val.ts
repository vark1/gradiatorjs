import { assert } from '../utils/utils';
import { Val } from './val';

export function random(shape: number[]) {
    let x = new Val(shape)
    return x
}

export function reshape(a: Val, shape: number[]) {

}

export function transpose(a: Val) : Val{
    if(a.dim === 1) return a;
    assert(a.dim === 2, () => 'transpose only supports 2D arrays');
    let x = new Float64Array(a.size)
    let y = a.data
    let newShape = [...a.shape];
    let res = new Val(newShape)
    newShape[0] = a.shape[1];
    newShape[1] = a.shape[0];

    for (let i=0; i<a.shape[0]; i++) {
        for (let j=0; j<a.shape[1]; j++) {
            x[j*a.shape[0] + i] = y[i*a.shape[1] + j]
        }
    }
    res.data=x
    return res
}