import { activationfn } from '../activations'
import {load_dataset} from '../utils/utils_data'
import { reshape, transpose } from '../Val/utils_val'
import * as ops from '../Val/ops'
import { Val } from '../Val/val'

// let [train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes] = load_dataset()

// let m_train = train_set_x_orig.length
// let m_test = test_set_x_orig.length
// let num_px = train_set_x_orig[0].length

// let train_set_x_flatten = transpose(reshape(train_set_x_orig, [train_set_x_orig.shape[0], -1]))
// let test_set_x_flatten = transpose(reshape(test_set_x_orig, [test_set_x_orig.shape[0], -1]))

// let train_set_x = ops.div(train_set_x_flatten, 255)
// let test_set_x = ops.div(test_set_x_flatten, 255)

function propagate(w: Val, b: Val, X: Val, Y: Val) {
    let m = X.shape[1]
    let z = ops.add(ops.dot(transpose(w), X), b)
    let A = activationfn(z, 'sigmoid')

    // cost = -1/m * (sum(Y*log(A) + (1-Y)*log(1-A)))
    let cost = ops.mul(-1/m, ops.sum(ops.add(ops.mul(Y,ops.log(A)), ops.mul(ops.sub(1, Y), ops.log(ops.sub(1, A))))))

    // backprop
    let dw = ops.mul(ops.dot(X, transpose(ops.sub(A, Y))), 1/m)     // dw = 1/m * (X.(A-Y).T)
    let db = ops.mul(ops.sum(ops.sub(A,Y)), 1/m)                    // db = 1/m * sum(A-Y)

    let grads = {dw, db}
    return [grads, cost]
}

let w = new Val([2,1]) 
w.data = [[1], [2]]
let b = new Val ([1], 1.5)
let X = new Val([2,3])
X.data= [[1., -2., -1.], [3., 0.5, -3.2]]
let Y = new Val([1,3])
Y.data = [[1, 1, 0]]

let [grads, cost] = propagate(w, b, X, Y)

console.log(grads['dw']) // [[ 0.25071532], [-0.06604096]]
console.log(grads['db']) // -0.1250040450043965
console.log(cost) // 0.15900537707692405


function optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=false) {
    
}