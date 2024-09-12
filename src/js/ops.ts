import { Tensor } from './tensor';
import { NDarr, t_any } from './types';
import * as utils from './utils';


// Scalar ops (tensor with a scalar)

export function pow<T extends number[]>(t: Tensor<T>, num: number) : Tensor<T> {
    const data = utils.ophelper_(t.data, '**', num)
    return new Tensor(data, t.shape, `(${t.label})^${num}`, '**')
}

export function sDiv<T extends number[]>(t: Tensor<T>, num: number) : Tensor<T> {
    const data = utils.ophelper_(t.data, '/', num)
    return new Tensor(data, t.shape, `(${t.label})/${num}`, '/')
}

//unary ops
export function negate<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, '*', -1)
    return new Tensor(data, t.shape, `(-${t.label})`, '¬')
}

//unary functions
export function exp<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, 'exp')
    return new Tensor(data, t.shape, `e^(${t.label})`, 'exp')
}

export function tanh<T extends number[]>(t: Tensor<T>) : Tensor<T> {
    const data = utils.ophelper_(t.data, 'tanh')
    return new Tensor(data, t.shape, `tanh(${t.label})`, 'tanh')
}

//binary ops
export function add<T extends number[]>(t1: t_any<T>, t2: t_any<T>) : Tensor<T> {
    [t1, t2] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1.rank === t2.rank, ()=> `In addition: Both tensors must have the same rank. got t1 rank: ${t1.rank} and t2 rank: ${t2.rank}`)
    utils.assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In addition: Both tensors must have the same shape')

    let additionResult = utils.addNDarrays(t1.data, t2.data)    
    let out = new Tensor(additionResult, t1.shape, `${t1.label} + ${t2.label}`, '+', [t1, t2])
    return out
}

export function subtract<T extends number[]>(t1: t_any<T>, t2: t_any<T>) : Tensor<T> {
    [t1, t2] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1.rank === t2.rank, ()=> `In subtraction: Both tensors must have the same rank. got t1 rank: ${t1.rank} and t2 rank: ${t2.rank}`)
    utils.assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In subtraction: Both tensors must have the same shape')

    let t2_ = negate(t2)
    let subtractionResult = utils.addNDarrays(t1.data, t2_.data)
    return new Tensor(subtractionResult, t1.shape, `${t1.label} - ${t2.label}`, '-')
}

export function dot<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
    utils.assert((t1.rank === 1 || t1.rank === 2) && (t2.rank === 1 || t2.rank === 2), () => `In dot: Both inputs must all be rank 1 or 2`);
    const t1Inner = (t1.rank === 1 ? t1.size : t1.shape[1]);
    const t2Inner = (t2.rank === 1 ? t2.size : t2.shape[0]);    
    utils.assert(t1Inner === t2Inner, ()=> `In dot: inner dimensions didn't match`)

    let shape : number[];
    let result : number[] | number[][] = [];
    if (t1.rank === 1 && t2.rank === 1) {
        // (1, x) * (x, 1) => (1, 1)
        let sum = 0;
        for(let i=0; i<t1.shape[0]; i++) {
            sum+=t1.data[i]*t2.data[i];
        }
        result = [sum];
        shape = [1];
        
    } else if (t1.rank === 1 && t2.rank === 2) {
        // (1, x) * (x, y) => (1, y)
        let res: number[] = [];
        for (let k=0; k<t2.shape[1]; k++) {
            let sum = 0;
            for (let i=0; i<t1.shape[0]; i++) {
                sum += t1.data[i] * t2.data[k][i];
            }
            res.push(sum);
        }
        result = res;
        shape = [t2.shape[1]];

    } else if (t1.rank === 2 && t2.rank === 1) {
        // (x, y) * (y, 1) => (x, 1)
        let res: number[] = [];
        for (let k=0; k<t1.shape[0]; k++) {
            let sum = 0;
            for (let i=0; i<t2.shape[0]; i++) {
                sum += t1.data[k][i] * t2.data[i]
            }
            res.push(sum)
        }
        result = res
        shape = [t1.shape[0]]

    } else if (t1.rank === 2 && t2.rank === 2) {
        // (x, y) * (y, z) (x, z)
        let res: number[][] = []
        for (let j=0; j<t1.shape[0]; j++) {
            let row: number[] = []
            for (let k=0; k<t2.shape[1]; k++) {
                let sum = 0;
                for (let i=0; i<t1.shape[1]; i++) {
                    sum += t1.data[j][i] * t2.data[i][k]
                }
                row.push(sum)
            }
            res.push(row)
        }
        result = res
        shape = [t1.shape[0], t2.shape[1]]
    }
    return new Tensor(result as NDarr<T>, shape, `${t1.label} . ${t2.label}`, 'dot')
}

export function multiply<T extends number[]>(t1: Tensor<T>, t2: Tensor<T>) : Tensor<T> {
    [t1, t2] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1.rank === t2.rank, ()=> `In hadamard product: Both tensors must have the same rank. got t1 rank: ${t1.rank} and t2 rank: ${t2.rank}`)
    utils.assert(t1.shape.every((dimension, index) => dimension == t2.shape[index]), () => 'In hadamard product: Both tensors must have the same shape')

    const result = utils.hadamardNDarrays(t1.data, t2.data)
    return new Tensor(result, t1.shape, `${t1.label} # ${t2.label}`, 'element wise multiplication')
}

export function backprop<T extends number[]>(t: Tensor<T>) {
    
}