import { DATASET_HDF5_TEST, DATASET_HDF5_TRAIN, catvnoncat_prepareDataset } from '../../src/js/utils/utils_datasets.js'
import { Val } from '../../src/val.js'
import * as ops from '../../src/ops.js'
import { assert } from '../../src/utils.js'
import { relu, sigmoid, tanh } from '../../src/activations.js'

interface dynamicObject {
    [key: string]: Val
}

interface cacheObject {A: Val, W: Val, b: Val}

export function initializeParams(layer_dims: number[]) {
    let parameters : dynamicObject = {}
    let L = layer_dims.length
    
    for (let l=1; l<L; l++) {
        parameters['W' + l.toString()] = ops.mul(new Val([layer_dims[l], layer_dims[l-1]]).randn(), 0.01)
        parameters['b' + l.toString()] = ops.mul(new Val([layer_dims[l], 1]), 0.01)
        
        assert(JSON.stringify(parameters['W' + l.toString()].shape) == JSON.stringify([layer_dims[l], layer_dims[l - 1]]), ()=>'w shape incorrect')
        assert(JSON.stringify(parameters['b' + l.toString()].shape) == JSON.stringify([layer_dims[l], 1]), ()=>'b shape incorrect')

    }
    return parameters
}

export function linearForward(A: Val, W: Val, b: Val): [Val, cacheObject]{
    let Z = ops.add(ops.dot(W, A), b)
    let cache = {A, W, b}
    return [Z, cache]
}

export function linearActivationForward(
    A_prev: Val, 
    W: Val, 
    b: Val, 
    activation: "relu" | "sigmoid" | "tanh"
): [Val, [cacheObject, Val]]{
    let [Z, linear_cache] = linearForward(A_prev, W, b)
    let A: Val
    let activation_cache = Z.clone()
    if (activation === 'relu') {
        A = relu(Z)
    }else if (activation === 'sigmoid') {
        A = sigmoid(Z)
    }else {
        A = tanh(Z)
    }    
    let cache: [cacheObject, Val] = [linear_cache, activation_cache]

    return [A, cache]
}

export function LModelForward(X: Val, parameters: dynamicObject) : [Val, [cacheObject, Val][]] {
    let caches = []
    let A = X.clone()
    let num_layers = Math.floor(Object.keys(parameters).length/2)

    for (let l=1; l<num_layers; l++) {
        let A_prev = A.clone()
        let cache: [cacheObject, Val]
        [A, cache] = linearActivationForward(
            A_prev, 
            parameters['W' + l.toString()], 
            parameters['b' + l.toString()], 
            'relu'
        )
        caches.push(cache)
    }

    let [AL, cache] = linearActivationForward(
        A, 
        parameters['W' + num_layers.toString()], 
        parameters['b' + num_layers.toString()], 
        'sigmoid'
    )
    caches.push(cache)

    return [AL, caches]
}

export function computeCost(AL: Val, Y: Val) {
    let m = Y.shape[1]
    let cost = ops.mul(-1/m, ops.sum(ops.add(ops.mul(Y, ops.log(AL)), ops.mul(ops.sub(1, Y), ops.log(ops.sub(1, AL))))))
    // TODO: squeeze the cost, to make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost
}

export function linearBackward(dZ: Val, cache: cacheObject) {
    let {A, W, b} = cache
    let A_prev = A
    let m = A_prev.shape[1]
    let dW = ops.mul(1/m, ops.dot(dZ, A_prev.T))
    let db = ops.mul(1/m, ops.sum(dZ, 1, true))
    let dA_prev = ops.dot(W.T, dZ)
    
    return [dA_prev, dW, db]
}

export function linearActivationBackward(dA: Val, cache: [cacheObject, Val], activation: "sigmoid" | "relu" ): Val[] {
    let [linear_cache, activation_cache] = cache
    let dZ, dA_prev, dW, db
    if(activation === "relu") {
        dZ = reluBackward(dA, activation_cache);
        [dA_prev, dW, db] = linearBackward(dZ, linear_cache)
    } else if(activation === "sigmoid") {
        dZ = sigmoidBackward(dA, activation_cache);
        [dA_prev, dW, db] = linearBackward(dZ, linear_cache)
    } else {
        throw new Error("Invalid activation type")
    }
    return [dA_prev, dW, db]
}

export function LModelBackward(AL: Val, Y: Val, caches: [cacheObject, Val][]) {
    let grads : dynamicObject = {}
    let L = caches.length
    let m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    let dAL = ops.negate(ops.sub(ops.divElementWise(Y, AL), ops.divElementWise(ops.sub(1,Y), ops.sub(1, AL))))

    let current_cache: [cacheObject, Val] = caches[L-1]
    let [dA_prev_temp, dW_temp, db_temp] = linearActivationBackward(dAL, current_cache, 'sigmoid')
    grads['dA' + (L-1).toString()] = dA_prev_temp;
    grads['dW' + (L).toString()] = dW_temp;
    grads['db' + (L).toString()] = db_temp;
    
    for (let l=L-2; l>=0; l--) {
        current_cache = caches[l];
        [dA_prev_temp, dW_temp, db_temp] = linearActivationBackward(dA_prev_temp, current_cache, 'relu');
        grads['dA' + (l).toString()] = dA_prev_temp;
        grads['dW' + (l+1).toString()] = dW_temp;
        grads['db' + (l+1).toString()] = db_temp;
    }
    return grads
}

export function updateParameters(params: dynamicObject, grads: dynamicObject, learning_rate: number): dynamicObject {
    let parameters = { ...params };
    let num_layers = Math.floor(Object.keys(parameters).length/2)

    for (let l=0; l<num_layers; l++) {
        parameters['W' + (l+1).toString()] = ops.sub(parameters['W' + (l+1).toString()], ops.mul(learning_rate, grads['dW' + (l+1).toString()]));
        parameters['b' + (l+1).toString()] = ops.sub(parameters['b' + (l+1).toString()], ops.mul(learning_rate, grads['db' + (l+1).toString()]));
    }
    return parameters
}

export function MLP(
    X: Val, 
    Y: Val, 
    layers_dims: number[], 
    learning_rate: number = 0.0075, 
    num_iterations: number = 3000, 
    print_cost: boolean = false
){
    let grads = {}
    let costs = []
    let m = X.shape[1]
    let [n_x, n_h, n_y] = layers_dims
    let parameters = initializeParams(layers_dims)

    for (let i=0; i<num_iterations; i++) {
        let [AL, caches] = LModelForward(X, parameters)
        let cost = computeCost(AL, Y)
        grads = LModelBackward(AL, Y, caches)
        parameters = updateParameters(parameters, grads, learning_rate)

        if (print_cost && (i%100 === 0) || (i === (num_iterations-1)))
            console.log(`Cost after iteration ${i}: ${cost.data}`)
        if (i%100 === 0 || (i===num_iterations))
            costs.push(cost)
    }
    return [parameters, costs]

}

const button = document.getElementById('run_model_btn');
if (button) {
    button.addEventListener('click', function() {
        if(DATASET_HDF5_TEST && DATASET_HDF5_TRAIN) {
            let [train_x, train_y, test_x, test_y] = catvnoncat_prepareDataset()
            let n_x = 12288
            let n_h = 7
            let n_y = 1
            let learning_rate = 0.0075
            let L_layer_model = MLP(train_x, train_y, [n_x, n_h, n_y], learning_rate, 2500, true)
            console.log(L_layer_model)
        }
    });
}

// backward propagation for as single RELU unit
function reluBackward (dA: Val, cache: Val) : Val {
    let Z = cache
    let dZ = dA.clone()
    for(let i=0; i<Z.size; i++) {
        if (Z.data[i]<=0) {
            dZ.data[i] = 0
        }
    }
    return dZ

    // let x = new Val(dA.shape)
    // x.data = dA.data.map((k: number) => ((k === 0 ? 0 : 1)))
    // return [x, dA]
}

// backward propagation for a single sigmoid unit
function sigmoidBackward(dA: Val, cache: Val) : Val {
    let Z = cache
    let s = ops.pow(ops.add(1, ops.exp(ops.negate(Z))), -1)
    let dZ = ops.mul(ops.mul(dA, s), ops.sub(1, s))
    return dZ
    // let x = new Val(dA.shape)
    // x.data = dA.data.map((k:number)=> (k * (1-k)))
    // return [x, dA]
}


/*
function MLP(
    train_x: Val,
    train_y: Val,
    learning_rate: number,
    iterations: number,
    print_cost: boolean
) {
    const { layerSizes, activations } = config;
    
    // Initialize parameters
    const parameters = initializeParameters(layerSizes);
    
    // Training loop
    for (let i = 0; i < iterations; i++) {
        // Forward propagation with activations
        const [AL, caches] = modelForward(train_x, parameters, activations);
        
        // Compute cost
        const cost = computeCost(AL, train_y);
        
        // Backward propagation
        const grads = modelBackward(AL, train_y, caches, activations);
        
        // Update parameters
        parameters = updateParameters(parameters, grads, learning_rate);
        
        if (print_cost && i % 100 === 0) {
            console.log(`Cost after iteration ${i}: ${cost.data[0]}`);
        }
    }
    return parameters;
}

function modelForward(
    X: Val,
    parameters: any,
    activations: string[]
): [Val, any[]] {
    const caches = [];
    let A = X;
    
    for (let l = 1; l < activations.length; l++) {
        const [A_prev, W, b] = [A, parameters[`W${l}`], parameters[`b${l}`]];
        const Z = ops.add(ops.matmul(W, A_prev), b);
        A = applyActivation(Z, activations[l-1]);
        caches.push([A_prev, W, b, Z]);
    }
    
    return [A, caches];
}

function applyActivation(Z: Val, activation: string): Val {
    switch (activation) {
        case 'relu':
            return ops.map(Z, x => relu(x));
        case 'sigmoid':
            return ops.map(Z, x => sigmoid(x));
        case 'tanh':
            return ops.map(Z, x => tanh(x));
        default:
            throw new Error(`Unsupported activation: ${activation}`);
    }
}
*/