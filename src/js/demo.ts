import {Value} from "./engine"

let a = new Value (2.0, 'a')
let b = new Value (-3.0, 'b')
let c = new Value (10, 'c')
let e = a.mul(b); e.label = 'e'
let d = e.add(c); d.label = 'd'
let f = new Value(-2.0, 'f')
let L = d.mul(f); L.label = 'L'
console.log(L.toString())