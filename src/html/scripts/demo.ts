import { Val } from "../../val.js";
import * as op from "../../ops.js"  // Assuming you have ops.ts with add, mul, etc.
import { tanh } from "../../activations.js"

/* 
--------------------------------------------------------------------------------------------------------
scalar network
--------------------------------------------------------------------------------------------------------
*/

// Create values with requiresGrad = true
const x1 = new Val([], 2.0);  // Scalar
const w1 = new Val([], -3.0);
const x2 = new Val([], 0.0);
const w2 = new Val([], 1.0);

const x1w1 = op.mul(x1, w1);
const x2w2 = op.mul(x2, w2);
const x1w1x2w2 = op.add(x1w1, x2w2);

const b = new Val([], 6.881373587019532);
const n = op.add(x1w1x2w2, b);
const o = tanh(n);

console.log("Forward pass:");
console.log("Output:", o.data[0]);

o.backward();

console.log("Backward pass:");

console.log("x1:", x1.data[0], x1.grad[0]);         // 2, -1.5
console.log("w1:", w1.data[0], w1.grad[0]);         // -3, 1.0
console.log("x2:", x2.data[0], x2.grad[0]);         // 0, 0.5
console.log("w2:", w2.data[0], w2.grad[0]);         // 1, 0.0

console.log("x1w1: ", x1w1.data[0], x1w1.grad[0])   // -6, 0.5
console.log("x2w2: ", x2w2.data[0], x2w2.grad[0])   // 0, 0.5

console.log("x1w1 + x2w2: ", x1w1x2w2.data[0], x1w1x2w2.grad[0])   // -6, 0.5

console.log("b:", b.data[0], b.grad[0]);            // 6.8814, 0.5

console.log("n [x1w1+x2w2 + b]:", n.data[0], n.grad[0]);            // 0.8814, 0.5
console.log("o [tanh(n)]:", o.data[0], o.grad[0]);            // 0.7071, 1

/* 
--------------------------------------------------------------------------------------------------------
logistic regression
--------------------------------------------------------------------------------------------------------
*/


/* 
--------------------------------------------------------------------------------------------------------
deep nn
--------------------------------------------------------------------------------------------------------
*/