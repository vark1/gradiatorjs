import {Value} from "./engine"

// let a = new Value (2.0, 'a')
// let b = new Value (-3.0, 'b')
// let c = new Value (10, 'c')
// let e = a.mul(b); e.label = 'e'
// let d = e.add(c); d.label = 'd'
// let f = new Value(-2.0, 'f')
// let L = d.mul(f); L.label = 'L'
// console.log(L.toString())

let x1 = new Value(2.0, 'x1')
let x2 = new Value(0.0, 'x2')
let w1 = new Value(-3.0, 'w1')
let w2 = new Value(1.0, 'w2')

let b = new Value(6.881373587019532, 'b')

let x1w1 = x1.mul(w1); x1w1.label = 'x1*w1'
let x2w2 = x2.mul(w2); x2w2.label = 'x2*w2'
let x1w1x2w2 = x1w1.add(x2w2); x1w1x2w2.label = 'x1*w1 + x2*w2'
let n = x1w1x2w2.add(b); n.label = 'n'
let o = n.tanh(); o.label = 'o'

o.backward()