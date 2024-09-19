import { Tensor } from './tensor';
import { t_any } from './types';
import * as utl from './utils/utils_nd';
import { broadcastAndConvertNum, convertToTensor } from './utils/utils_tensor';
import {assert} from './utils/utils'

// TODO : ADD BACKWARD FOR REMAINING OPS; OPTIMIZE USING TYPED ARRAYS AND DEL EXCESS TENSOR CREATIONS 
// MAYBE CONVERT NDARR TO A SINGLE TYPED ARRAY FOR PERF? 

export function pow(t: t_any, num: number) : Tensor {
    let t_ = convertToTensor(t)
    const data = utl.applyFnUnary(t_.data, (x)=> (x ** num))
    let out = new Tensor(data, `(${t_.label})^${num}`, t_.shape, '**', [t_])

    out._backward = () => {
        let derivative = mul(mul(num, pow(t_, (num-1))), out.grad)
        t_.grad = add(t_.grad, derivative).data;
    };
    return out
}

export function div(t: t_any, num: number) : Tensor {
    let t_ = convertToTensor(t)
    const data = utl.applyFnUnary(t_.data, (x)=> (x / num))
    return new Tensor(data, `(${t_.label})/${num}`, t_.shape, '/', [t_])
}

//unary ops
export function negate(t: t_any) : Tensor {
    const t_ = mul(t, -1)
    t_.label = `(-${t_.label})`
    t_.op = '¬'
    t_._prev = [t_]
    return t_;
}

export function exp(t: t_any) : Tensor {
    let t_ = convertToTensor(t)
    const data = utl.applyFnUnary(t_.data, (x)=> (Math.exp(x)))
    let out = new Tensor(data, `e^(${t_.label})`, t_.shape, 'exp', [t_])

    out._backward = () => {
        let derivative = mul(out.data, out.grad)
        t_.grad = add(t_.grad, derivative).data;
    };

    return out
}

//binary ops
export function add(t1: t_any, t2: t_any) : Tensor {
    let [t1_, t2_] = broadcastAndConvertNum(t1, t2)
    assert(t1_.rank === t2_.rank, ()=> `In addition: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In addition: Both tensors must have the same shape')

    let additionResult = utl.applyFnBinary(t1_.data, t2_.data, (x, y)=>(x + y))    
    let out = new Tensor(additionResult, `${t1_.label} + ${t2_.label}`, t1_.shape, '+', [t1_, t2_])
    out._backward = () => {
        t1_.grad = add(t1_.grad, out.grad).data
        t2_.grad = add(t2_.grad, out.grad).data
    }
    return out
}

export function sub(t1: t_any, t2: t_any) : Tensor {
    let [t1_, t2_] = broadcastAndConvertNum(t1, t2)
    assert(t1_.rank === t2_.rank, ()=> `In subtraction: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In subtraction: Both tensors must have the same shape')

    let subtractionResult = utl.applyFnBinary(t1_.data, t2_.data, (x,y)=>(x-y))
    let out = new Tensor(subtractionResult, `${t1_.label} - ${t2_.label}`, t1_.shape, '-', [t1_, t2_])
    return out
}

export function dot(t1: t_any, t2: t_any) : Tensor {
    t1 = convertToTensor(t1)
    t2 = convertToTensor(t2)
    assert((t1.rank === 1 || t1.rank === 2) && (t2.rank === 1 || t2.rank === 2), () => `In dot: Both inputs must all be rank 1 or 2`);
    const t1Inner = (t1.rank === 1 ? t1.size : t1.shape[1]);
    const t2Inner = (t2.rank === 1 ? t2.size : t2.shape[0]);    
    assert(t1Inner === t2Inner, ()=> `In dot: inner dimensions didn't match`)

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
    return new Tensor(result, `${t1.label} . ${t2.label}`, shape, 'dot', [t1, t2])
}

export function mul(t1: t_any, t2: t_any) : Tensor {
    let [t1_, t2_] = broadcastAndConvertNum(t1, t2)
    assert(t1_.rank === t2_.rank, ()=> `In hadamard product: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In hadamard product: Both tensors must have the same shape')

    const result = utl.applyFnBinary(t1_.data, t2_.data, (x,y)=>(x*y))

    let out = new Tensor(result, `${t1_.label} # ${t2_.label}`, t1_.shape, 'element wise multiplication', [t1_, t2_])
    out._backward = () => {
        t1_.grad = add(t1_.grad, mul(t2_, out.grad)).data
        t2_.grad = add(t2_.grad, mul(t1_, out.grad)).data
    }
    return out
}