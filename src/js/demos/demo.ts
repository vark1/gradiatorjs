// import {Tensor} from '../nd_old/tensor';
// import * as op from '../Val/ops';
// import * as act from '../activations'

// //scalar
// // let x = new Tensor(1, 'x')
// // let y = new Tensor(2, 'y')
// // let z = op.add(x, y); z.label = 'z'
// // console.log(z)

// // let a = new Tensor (2.0, 'a')
// // let b = new Tensor (-3.0, 'b')
// // let c = new Tensor (10, 'c')
// // let e = op.multiply(a, b); e.label = 'e'
// // let d = op.add(e, c); d.label = 'd'
// // let f = new Tensor(-2.0, 'f')
// // let L = op.multiply(d, f); L.label = 'L'
// // console.log(L)

// let x1 = new Tensor(2.0, 'x1')
// let x2 = new Tensor(0.0, 'x2')
// let w1 = new Tensor(-3.0, 'w1')
// let w2 = new Tensor(1.0, 'w2')

// let b = new Tensor(6.881373587019532, 'b')

// let x1w1 = op.mul(x1, w1); x1w1.label = 'x1*w1'
// let x2w2 = op.mul(x2, w2); x2w2.label = 'x2*w2'
// let x1w1x2w2 = op.add(x1w1, x2w2); x1w1x2w2.label = 'x1*w1 + x2*w2'
// let n = op.add(x1w1x2w2, b); n.label = 'n'
// let o = act.activationfn(n, 'tanh'); o.label = 'o'
// o.backpropagation()

// // x1.print()
// // w1.print()
// // x2.print()
// // w2.print()
// // x1w1.print()
// // x2w2.print()
// // x1w1x2w2.print()
// // b.print()
// // n.print()
// // o.print()