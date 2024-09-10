import { Tensor } from './tensor';
import * as utils from './utils';


// Scalar ops (1 tensor and 1 scalar)
export function mul<T extends number[]>(t: Tensor<T>, num: number) : Tensor<T> {
    const data = utils.ophelper_(t.data, '*', num)
    return new Tensor(data, t.shape, `${num}*(${t.label})`)
}

export function pow<T extends number[]>(t: Tensor<T>, num: number) : Tensor<T> {
    const data = utils.ophelper_(t.data, '**', num)
    return new Tensor(data, t.shape, `(${t.label})^${num}`)
}

//unary ops
export function negate<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, '*', -1)
    return new Tensor(data, t.shape, `(-${t.label})`)
}

//unary functions
export function exp<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, 'exp')
    return new Tensor(data, t.shape, `e^(${t.label})`)
}

export function tanh<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, 'tanh')
    return new Tensor(data, t.shape, `tanh(${t.label})`)
}

//binary ops/functions

export function add<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
    utils.assert(t1.rank === t2.rank, ()=> `In addition: Both tensors must have the same rank. got t1 rank: ${t1.rank} and t2 rank: ${t2.rank}`)
    utils.assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In addition: Both tensors must have the same shape')

    let additionResult = utils.addNDarrays(t1.data, t2.data)
    return new Tensor(additionResult, t1.shape, `${t1.label} + ${t2.label}`, '+')
}

// export function dot<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
//     utils.assert((t1.rank === 1 || t1.rank === 2) && (t2.rank === 1 || t2.rank === 2), () => `In mul: Both inputs must all be rank 1 or 2`);
//     const t1Inner = (t1.rank === 1 ? t1.size : t1.shape[1]);
//     const t2Inner = (t2.rank === 1 ? t2.size : t2.shape[0]);    

//     //matmul on rank basis
    
// }