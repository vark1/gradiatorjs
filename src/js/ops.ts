import { Tensor } from './tensor';
import * as utils from './utils';

export function add<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
    utils.assert(t1.rank !== t2.rank, ()=> "In addition: Both tensors must have the same rank")
    utils.assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In addition: Both tensors must have the same shape')

    let additionResult = utils.addNDarrays(t1.data, t2.data)
    return new Tensor(additionResult, t1.shape, `${t1.label} + ${t2.label}`)
}

export function dot<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
    utils.assert((t1.rank === 1 || t1.rank === 2) && (t2.rank === 1 || t2.rank === 2), () => `In mul: Both inputs must all be rank 1 or 2`);


}