import { Tensor } from './tensor';
import { NDArray } from './types';
import * as utils from './utils';


export function pow(t: Tensor, num: number) : Tensor {
    const data = utils.ophelper_(t.data, '**', num)
    let out = new Tensor(data, `(${t.label})^${num}`, t.shape, '**', [t])
    function _backward() {
        t.grad = utils.addNDarrays(t.grad, multiply(multiply(num, utils.ophelper_(t.data, '**', (num-1))), out.grad).data)
    }
    out._backward = _backward
    return out
}

export function sDiv(t: Tensor, num: number) : Tensor {
    const data = utils.ophelper_(t.data, '/', num)
    return new Tensor(data, `(${t.label})/${num}`, t.shape, '/', [t])
}

//unary ops
export function negate(t: Tensor) : Tensor {
    const t_ = multiply(t, -1)
    t_.label = `(-${t.label})`
    t.op = '¬'
    t._prev = [t]
    return t_;
}

//unary functions
export function exp(t: Tensor) : Tensor {
    const data = utils.ophelper_(t.data, 'exp')
    let out = new Tensor(data, `e^(${t.label})`, t.shape, 'exp', [t])
    
    function _backward() {
        t.grad = add(t.grad, multiply(out.data, out.grad)).data
    }
    out._backward = _backward

    return out
}

export function tanh(t: Tensor) : Tensor {
    const data = utils.ophelper_(t.data, 'tanh')
    let out = new Tensor(data, `tanh(${t.label})`, t.shape, 'tanh', [t])
    
    function _backward() { 
        t.grad = add(t.grad, multiply(subtract(1, pow(t, 2)), out.grad).data).data
        // console.log(1-((0.7071)**2))
    }
    out._backward = _backward

    return out
}

//binary ops
export function add(t1: NDArray | Tensor, t2: NDArray | Tensor) : Tensor {
    let [t1_, t2_] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1_.rank === t2_.rank, ()=> `In addition: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    utils.assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In addition: Both tensors must have the same shape')

    let additionResult = utils.addNDarrays(t1_.data, t2_.data)    
    let out = new Tensor(additionResult, `${t1_.label} + ${t2_.label}`, t1_.shape, '+', [t1_, t2_])
    let backward = () => {
        t1_.grad = utils.addNDarrays(t1_.grad, out.grad)
        t2_.grad = utils.addNDarrays(t2_.grad, out.grad)
    }
    out._backward = backward
    return out
}

export function subtract(t1: NDArray | Tensor, t2: NDArray | Tensor) : Tensor {
    let [t1_, t2_] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1_.rank === t2_.rank, ()=> `In subtraction: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    utils.assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In subtraction: Both tensors must have the same shape')

    let t2__ = negate(t2_)
    let subtractionResult = utils.addNDarrays(t1_.data, t2__.data)
    let out = new Tensor(subtractionResult, `${t1_.label} - ${t2_.label}`, t1_.shape, '-', [t1_, t2_])
    let backward = () => {
        t1_.grad = utils.addNDarrays(t1_.grad, out.grad)
        t2_.grad = utils.addNDarrays(t2_.grad, out.grad)
    }
    out._backward = backward
    return out
}

export function dot(t1: Tensor, t2: Tensor) : Tensor {
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
    return new Tensor(result, `${t1.label} . ${t2.label}`, shape, 'dot', [t1, t2])
}

export function multiply(t1: NDArray | Tensor, t2: NDArray | Tensor) : Tensor {
    let [t1_, t2_] = utils.broadcastAndConvertNum(t1, t2)
    utils.assert(t1_.rank === t2_.rank, ()=> `In hadamard product: Both tensors must have the same rank. got t1_ rank: ${t1_.rank} and t2_ rank: ${t2_.rank}`)
    utils.assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In hadamard product: Both tensors must have the same shape')

    const result = utils.hadamardNDarrays(t1_.data, t2_.data)

    let out = new Tensor(result, `${t1_.label} # ${t2_.label}`, t1_.shape, 'element wise multiplication', [t1_, t2_])
    function _backward() {
        t1_.grad = utils.addNDarrays(t1_.grad, utils.hadamardNDarrays(t2_.data, out.grad))
        t2_.grad = utils.addNDarrays(t2_.grad, utils.hadamardNDarrays(t1_.data, out.grad)) 
    }
    out._backward = _backward
    return out
}