import { Val } from "../../src/val.js"

function Test() {
    let w = new Val([2,1]) 
    w.data = [[1], [2]]
    let b = new Val ([1], 1.5)
    let X = new Val([2,3])
    X.data= [[1., -2., -1.], [3., 0.5, -3.2]]
    let Y = new Val([1,3])
    Y.data = [[1, 1, 0]]
    
    // let [grads, cost] = propagate(w, b, X, Y)
    // console.log(grads['dw'].data) // [[ 0.25071532], [-0.06604096]]
    // console.log(grads['db'].data) // -0.1250040450043965
    // console.log(cost.data) // 0.15900537707692405

    // let [params, gradients, costs] = optimize(w, b, X, Y, 100, 0.009, false)
    // console.log(params['w'].data) // [[0.80956046], [2.0508202 ]]
    // console.log(params['b'].data) // 1.5948713189708588
    // console.log(gradients['dw'].data) // [[ 0.17860505], [-0.04840656]]
    // console.log(gradients['db'].data) // -0.08888460336847771
    // console.log(costs) // 0.15900538

    // let w = new Val([2, 1])
    // w.data = [[0.1124579], [0.23106775]]
    // let b = new Val([1], -0.3)
    // let X = new Val([2,3])
    // X.data = [[1., -1.1, -3.2],[1.2, 2., 0.1]]
    // console.log(predict(w, b, X))   // [[1, 1, 0]]
}
