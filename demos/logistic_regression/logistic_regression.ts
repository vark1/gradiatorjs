import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset } from '../../src/js/utils/utils_datasets.js';
import { Val } from '../../src/val.js'
import { sigmoid } from '../../src/activations.js'
import { add, mul, sum, log, dot, div, mean, abs, sub } from '../../src/ops.js'

function propagate(w: Val, b: Val, X: Val, Y: Val) : [{dw: Val; db: Val}, Val]{
    let m = X.shape[1]
    let z = add(dot(w.T, X), b)
    let A = sigmoid(z)

    // cost = -1/m * sum(Y*log(A) + (1-Y)*log(1-A))
    let cost = mul(-1/m, sum(add(mul(Y,log(A)), mul(sub(1, Y), log(sub(1, A))))))

    // backprop
    let dZ = sub(A, Y)
    let dw = mul(dot(X, dZ.T), 1/m)     // dw = 1/m * (X.(A-Y).T)
    let db = mul(sum(dZ), 1/m)          // db = 1/m * sum(A-Y)

    let grads = {dw, db}
    return [grads, cost]
}


function optimize(w: Val, b: Val, X: Val, Y: Val, num_iterations=100, learning_rate=0.009, print_cost=false) : [{w: Val, b: Val}, {dw: Val, db: Val}, Float64Array[]] {
    let w_ = w.clone()
    let b_ = b.clone()

    let dw = new Val([0])
    let db = new Val([0])

    let costs = []

    for (let i=0; i<num_iterations; i++) {
        let [grads, cost] = propagate(w_, b_, X, Y)
        dw = grads['dw']
        db = grads['db']

        w_ = sub(w_, mul(learning_rate, dw))
        b_ = sub(b_, mul(learning_rate, db))

        if (i % 100 === 0) {
            costs.push(cost.data)
            // Print the cost every 100 training iterations
            if (print_cost) {
                console.log(`cost after iteration ${i}: ${cost.data}`)
            }
        }
    }
    let params = {"w": w_, "b": b_}
    let gradients = {'dw': dw, 'db': db}
    return [params, gradients, costs]
}

function predict(w: Val, b: Val, X: Val) : Val {
    let m = X.shape[1]
    let Y_prediction = new Val([1, m])
    w = w.reshape([X.shape[0], 1])
    let A = sigmoid(add(dot(w.T, X), b))

    for (let i=0; i<A.shape[1]; i++) {
        if(A.data[i] > 0.5) {
            Y_prediction.data[i] = 1
        }else {
            Y_prediction.data[i] = 0
        }
    }
    return Y_prediction
}

export function model(X_train: Val, Y_train: Val, X_test: Val, Y_test: Val, iterations=2000, l_rate=0.5, print_cost=false) 
: {costs: Float64Array[], Y_prediction_test: Val, Y_prediction_train: Val, w: Val, b: Val, learning_rate: number, num_iterations: number} {
    let w = new Val([X_train.shape[0], 1])
    let b = new Val([1])

    let [params, grads, costs] = optimize(w, b, X_train, Y_train, iterations, l_rate, print_cost)
    w = params['w']
    b = params['b']
    let Y_prediction_train = predict(w, b, X_train)
    let Y_prediction_test = predict(w, b, X_test)

    if(print_cost) {
        console.log(`train accuracy: ${sub(100, mul(100, mean(abs(sub(Y_prediction_train, Y_train))))).data[0]}`)
        console.log(`test accuracy: ${sub(100, mul(100, mean(abs(sub(Y_prediction_test, Y_test))))).data[0]}`)
    }

    let res = {
        'costs': costs, 
        'Y_prediction_test': Y_prediction_test, 
        'Y_prediction_train': Y_prediction_train,
        'w': w,
        'b': b,
        'learning_rate': l_rate,
        'num_iterations': iterations
    }

    return res
}

const button = document.getElementById('run_model_btn');
if (button) {
    button.addEventListener('click', function() {
        if(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN) {
            let [train_x, train_y, test_x, test_y] = catvnoncat_prepareDataset()
            let logistic_regression_model = model(train_x, train_y, test_x, test_y, 2000, 0.005, true)
            console.log(logistic_regression_model)
        }
    });
}