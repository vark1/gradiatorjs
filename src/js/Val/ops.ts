import {assert} from '../utils/utils.js'
import { Val } from './val.js';

function broadcast(t1: Val|number, t2: Val|number) : [Val, Val] {
    //dim check to make sure we're only broadcasting when the other tensor is not a scalar tensor aswell
    let t1shape: number[] = []
    let t2shape: number[] = []
    let t1data = null
    let t2data = null
    if (typeof t1 === 'number' && t2 instanceof Val && t2.dim !== 0) {
        t1shape = t2.shape
        t1data = new Float64Array(t2.size).fill(t1)
    }else if (typeof t2 === 'number' && t1 instanceof Val && t1.dim !== 0) {
        t2shape = t1.shape
        t2data = new Float64Array(t1.size).fill(t2)
    }else if (t1 instanceof Val && t1.size === 1 && t2 instanceof Val) {
        t1shape = t2.shape
        t1data = new Float64Array(t2.size).fill(t1.data[0])
    }else if(t2 instanceof Val && t2.size === 1 && t1 instanceof Val) {
        t2shape = t1.shape
        t2data = new Float64Array(t1.size).fill(t2.data[0])
    }
    let t1_ = t1
    let t2_ = t2
    if (t1data) {
        t1_ = new Val(t1shape)
        t1_.data = t1data
    }
    if (t2data) {
        t2_ = new Val(t2shape)
        t2_.data = t2data
    }
    return [<Val>t1_, <Val>t2_]
}

export function add(t1: Val|number, t2: Val|number) : Val {
    let [t1_, t2_] = broadcast(t1, t2)
    assert(t1_.dim === t2_.dim, ()=> `In addition: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In addition: Both matrices must have the same shape')

    let res = new Val(t1_.shape)
    res.data = t1_.data.map((num: number, idx: number)=>num + t2_.data[idx])
    return res
}

export function sub(t1: Val|number, t2: Val|number) : Val {
    let [t1_, t2_] = broadcast(t1, t2)
    assert(t1_.dim === t2_.dim, ()=> `In subtraction: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In subtraction: Both matrices must have the same shape')

    let res = new Val(t1_.shape)
    res.data = t1_.data.map((num: number, idx: number)=>num - t2_.data[idx])
    return res
}

export function mul(t1: Val|number, t2: Val|number) : Val {
    let [t1_, t2_] = broadcast(t1, t2)
    assert(t1_.dim === t2_.dim, ()=> `In hadamard product: Both matrices must have the same dim. got t1_dim: ${t1_.dim} and t2_dim: ${t2_.dim}`)
    assert(t1_.shape.every((dimension, index) => dimension == t2_.shape[index]), () => 'In hadamard product: Both matrices must have the same shape')

    let res = new Val(t1_.shape)
    res.data = t1_.data.map((num: number, idx: number)=> num*t2_.data[idx])
    return res
}

export function dot(t1: Val, t2: Val) : Val {
    assert((t1.dim === 1 || t1.dim === 2) && (t2.dim === 1 || t2.dim === 2), () => `In dot: Both inputs must all be dim 1 or 2`);
    if (t1.dim === 1 && t2.dim === 1) {
        return sum(mul(t1, t2))
    }

    const t1Inner = (t1.dim === 1 ? t1.size : t1.shape[1]);
    const t2Inner = (t2.dim === 1 ? t2.size : t2.shape[0]);    
    assert(t1Inner === t2Inner, ()=> `In dot: inner dimensions didn't match. dimensioins got: ${t1Inner} and ${t2Inner}`)
    
    let t1_ = t1
    let t2_ = t2

    if (t1.dim === 1) {
        t1_ = new Val([1, t1.shape[0]])
        t1_.data = Float64Array.from(t1.data)
    }

    if (t2.dim === 1) {
        t2_ = new Val([t2.shape[0], 1])
        t2_.data = Float64Array.from(t2.data)
    }

    let size = t1_.shape[0]*t2_.shape[1]
    let shape = [t1_.shape[0], t2_.shape[1]]

    let res = new Float64Array(size)
    for (let i=0; i<size; i++) {
        let sum=0
        let r = Math.floor(i/t2_.shape[1])
        let c = Math.floor(i%t2_.shape[1])
        for (let j=0; j<t1_.shape[1]; j++) {
            sum += t1_.data[r * t1_.shape[1] + j] * t2_.data[j * t2_.shape[1] + c]
        }
        res[i] = sum
    }
    let final = new Val(shape)
    final.data = res
    return final
}

/* 
// old dot
export function dot(t1: Val, t2: Val) : Val {
    assert((t1.dim === 1 || t1.dim === 2) && (t2.dim === 1 || t2.dim === 2), () => `In dot: Both inputs must all be dim 1 or 2`);
    const t1Inner = (t1.dim === 1 ? t1.size : t1.shape[1]);
    const t2Inner = (t2.dim === 1 ? t2.size : t2.shape[0]);    
    assert(t1Inner === t2Inner, ()=> `In dot: inner dimensions didn't match. dimensioins got: ${t1Inner} and ${t2Inner}`)

    let shape : number[] = [];
    let result : number[] | number[][] | Float64Array = [];
    if (t1.shape[0] === 1 && t1.shape[1] === t2.shape[0] && t2.shape[1] === 1) {
        // (1, x) * (x, 1) => (1, 1)
        let sum = 0;
        for(let i=0; i<t1.shape[1]; i++) {
            sum+=t1.data[i]*t2.data[i];
        }
        result = [sum];
        shape = [1];
        
    } else if (t1.shape[0] === 1 && t1.shape[1] === t2.shape[0] && t2.shape[1] !== 1) {
        // (1, x) * (x, y) => (1, y)
        let res: number[] = [];
        for (let c=0; c<t2.shape[1]; c++) {
            let sum = 0;
            for (let r=0; r<t1.shape[1]; r++) {
                sum += t1.data[r] * t2.data[r * t2.shape[1] + c];
            }
            res.push(sum);
        }
        result = res;
        shape = [t2.shape[1]];

    } else if (t1.shape[0] !== 1 && t1.shape[1] === t2.shape[0] && t2.shape[1] === 1) {
        // (x, y) * (y, 1) => (x, 1)
        let res: number[] = [];
        for (let r=0; r<t1.shape[0]; r++) {
            let sum = 0;
            for (let c=0; c<t1.shape[1]; c++) {
                sum += t1.data[r * t1.shape[1] + c] * t2.data[c]
            }
            res.push(sum)
        }
        result = res
        shape = [t1.shape[0]]

    } else if (t1.shape[0] !== 1 && t1.shape[1] === t2.shape[0] && t2.shape[1] !== 1) {
        // (x, y) * (y, z) => (x, z)
        let size = t1.shape[0]*t2.shape[1]
        let res = new Float64Array(size)
        for (let i=0; i<size; i++) {
            let sum=0
            let r = Math.floor(i/t2.shape[1])
            let c = Math.floor(i%t2.shape[1])
            for (let j=0; j<t1.shape[1]; j++) {
                sum += t1.data[r * t1.shape[1] + j] * t2.data[j * t2.shape[1] + c]
            }
            res[i] = sum
        }
        result = res
        shape = [t1.shape[0], t2.shape[1]]
    }
    let final = new Val(shape)
    final.data = result
    return final
}
*/

export function pow(t: Val, num: number) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => k ** num)
    return x
}

export function div(t: Val, num: number) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => k / num)
    return x
}

export function negate(t: Val) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => -k)
    return x
}

export function abs(t: Val) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => Math.abs(k))
    return x
}

export function exp(t: Val) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => Math.exp(k))
    return x
}

export function log(t: Val) : Val {
    let x = new Val(t.shape)
    x.data = t.data.map((k:number) => Math.log(k))
    return x
}

export function sum(t: Val) : Val {
    let x = new Val([1])
    x.data = [t.data.reduce((a: number, c: number)=> a+c)]
    return x
}

export function mean(t: Val) : Val {
    let x = new Val([1])
    x.data = [t.data.reduce((a: number, c: number)=> a+c)/t.data.length]
    return x
}