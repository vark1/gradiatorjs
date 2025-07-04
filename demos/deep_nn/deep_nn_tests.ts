import { Val } from "../../src/val.js"
import * as fn from '../deep_nn/deep_nn.js'

// TODO: Change console.log to assert

interface dynamicObject {
    [key: string]: Val
}

interface cacheObject {A: Val, W: Val, b: Val}

// TESTS
function initializeParamsDeep_TEST() {
    let parameters = fn.initializeParams([5,4,3])
    console.log(parameters)
}

function linearForward_TEST() {
    let t_A = new Val([3, 2])
    let t_W = new Val([1, 3])
    let t_b = new Val([1, 1])
    t_A.data = [[ 1.62434536, -0.61175641], [-0.52817175, -1.07296862], [ 0.86540763, -2.3015387 ]]
    t_W.data = [[ 1.74481176, -0.7612069, 0.3190391 ]]
    t_b.data = [[-0.24937038]]
    
    let [Z, linear_cache] = fn.linearForward(t_A, t_W, t_b)
    console.log(Z)  // should be Z = [[ 3.26295337 -1.23429987]]
    console.log("________________________________________________________________")
    console.log(linear_cache)
}

function linearActivationForward_TEST() {
    let t_A_prev = new Val([3, 2])
    let t_W = new Val([1, 3])
    let t_b = new Val([1, 1])
    t_A_prev.data = [[-0.41675785, -0.05626683], [-2.1361961, 1.64027081], [-1.79343559, -0.84174737]]
    t_W.data = [[ 0.50288142, -1.24528809, -1.05795222]]
    t_b.data = [[-0.90900761]]

    let [t_A1, t_linear_activation_cache1] = fn.linearActivationForward(t_A_prev, t_W, t_b, "sigmoid")
    console.log("With sigmoid")
    console.log(t_A1)   // should be [[0.96890023 0.11013289]]
    
    console.log("________________________________________________________________")

    let [t_A2, t_linear_activation_cache2] = fn.linearActivationForward(t_A_prev, t_W, t_b, "relu")
    console.log("With relu")
    console.log(t_A2)   // should be [[3.43896131 0.        ]]
}

function LModelForward_TEST() {
    let X = new Val([5,4])
    let W1 = new Val([4,5])
    let b1 = new Val([4,1])
    let W2 = new Val([3,4])
    let b2 = new Val([3,1])
    let W3 = new Val([1,3])
    let b3 = new Val([1,1])

    X.data = [[-0.31178367,  0.72900392,  0.21782079, -0.8990918 ],
    [-2.48678065,  0.91325152,  1.12706373, -1.51409323],
    [ 1.63929108, -0.4298936,   2.63128056,  0.60182225],
    [-0.33588161,  1.23773784,  0.11112817,  0.12915125],
    [ 0.07612761, -0.15512816,  0.63422534,  0.810655  ]]

    W1.data = [[ 0.35480861,  1.81259031, -1.3564758 , -0.46363197,  0.82465384],
    [-1.17643148,  1.56448966,  0.71270509, -0.1810066 ,  0.53419953],
    [-0.58661296, -1.48185327,  0.85724762,  0.94309899,  0.11444143],
    [-0.02195668, -2.12714455, -0.83440747, -0.46550831,  0.23371059]]
    b1.data = [[ 1.38503523],
    [-0.51962709],
    [-0.78015214],
    [ 0.95560959]]

    W2.data = [[-0.12673638, -1.36861282,  1.21848065, -0.85750144],
    [-0.56147088, -1.0335199 ,  0.35877096,  1.07368134],
    [-0.37550472,  0.39636757, -0.47144628,  2.33660781]]
    b2.data = [[ 1.50278553],
    [-0.59545972],
    [ 0.52834106]]

    W3.data = [[ 0.9398248 ,  0.42628539, -0.75815703]]
    b3.data = [[-0.16236698]]

    let parameters = {"W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }

    let [AL, caches] = fn.LModelForward(X, parameters)
    console.log(AL) // should be AL = [[0.03921668 0.70498921 0.19734387 0.04728177]]
}

function computeCost_TEST() {
    let Y = new Val([1,3])
    Y.data = [[1, 1, 0]]
    let aL = new Val([1,3])
    aL.data = [[.8,.9,0.4]]
    
    console.log(fn.computeCost(aL, Y))
}

function linearBackward_TEST() {
    let dZ = new Val([3,4])
    let A = new Val([5,4])
    let W = new Val([3,5])
    let b = new Val([3,1])

    dZ.data = [[ 1.62434536, -0.61175641, -0.52817175, -1.07296862], 
    [ 0.86540763, -2.3015387,   1.74481176, -0.7612069 ],
    [ 0.3190391,  -0.24937038,  1.46210794, -2.06014071]]

    A.data = [[-0.3224172 , -0.38405435,  1.13376944, -1.09989127],
    [-0.17242821, -0.87785842,  0.04221375,  0.58281521],
    [-1.10061918,  1.14472371,  0.90159072,  0.50249434],
    [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
    [-0.26788808,  0.53035547, -0.69166075, -0.39675353]]

    W.data = [[-0.6871727 , -0.84520564, -0.67124613, -0.0126646 , -1.11731035],
    [ 0.2344157 ,  1.65980218,  0.74204416, -0.19183555, -0.88762896],
    [-0.74715829,  1.6924546 ,  0.05080775, -0.63699565,  0.19091548]]

    b.data = [[2.10025514],
    [0.12015895],
    [0.61720311]]

    let linear_cache = {A, W, b}

    let [dA_prev, dW, db] = fn.linearBackward(dZ, linear_cache)
    console.log(dA_prev)
    console.log(dW)
    console.log(db)
    /*
    should be 
    dA_prev: 
    [[-1.15171336  0.06718465 -0.3204696   2.09812712]
    [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
    [-0.4319552  -1.30987417  1.72354705  0.05070578]
    [-0.38981415  0.60811244 -1.25938424  1.47191593]
    [-2.52214926  2.67882552 -0.67947465  1.48119548]]
    dW: 
    [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]
    [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]
    [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]
    db: 
    [[-0.14713786]
    [-0.11313155]
    [-0.13209101]]
    */
}

function linearActivationBackward_TEST() {
    let dA = new Val([1,2])
    let A = new Val([3,2])
    let W = new Val([1,3])
    let b = new Val([1,1])
    let Z = new Val([1,2])

    dA.data = [[-0.41675785, -0.05626683]]
    
    A.data = [[-2.1361961 ,  1.64027081],
    [-1.79343559, -0.84174737],
    [ 0.50288142, -1.24528809]]

    W.data = [[-1.05795222, -0.90900761,  0.55145404]]

    b.data = [[2.29220801]]
    Z.data = [[ 0.04153939, -1.11792545]]
    
    let linear_cache : cacheObject = {A, W, b}
    let linear_activation_cache : [cacheObject, Val] = [linear_cache, Z]

    let [dA_prev_1, dW_1, db_1] = fn.linearActivationBackward(dA, linear_activation_cache, 'sigmoid')
    let [dA_prev_2, dW_2, db_2] = fn.linearActivationBackward(dA, linear_activation_cache, 'relu')

    console.log(dA_prev_1, dW_1, db_1)
    /*
    With sigmoid: 
    dA_prev = [[ 0.11017994  0.01105339]
    [ 0.09466817  0.00949723]
    [-0.05743092 -0.00576154]]
    dW = [[ 0.10266786  0.09778551 -0.01968084]]
    db = [[-0.05729622]]
    */
    
    console.log(dA_prev_2, dW_2, db_2)
    /*
    With relu: 
    dA_prev = [[ 0.44090989  0.        ]
    [ 0.37883606  0.        ]
    [-0.2298228   0.        ]]
    dW = [[ 0.44513824  0.37371418 -0.10478989]]
    db = [[-0.20837892]]
    */
}

function LModelBackward_TEST() {
    let AL = new Val([1, 2]);
    let Y = new Val([1, 2])
    
    let A1 = new Val([4,2]);
    let W1 = new Val([3,4]);
    let b1 = new Val([3,1]);
    let Z1 = new Val([3,2]);
    
    let A2 = new Val([3,2]);
    let W2 = new Val([1,3]);
    let b2 = new Val([1,1]);
    let Z2 = new Val([1,2]);

    AL.data = [[1.78862847, 0.43650985]]
    Y.data = [[1, 0]]

    A1.data = [[ 0.09649747, -1.8634927 ],
    [-0.2773882 , -0.35475898],
    [-0.08274148, -0.62700068],
    [-0.04381817, -0.47721803]]

    W1.data = [[-1.31386475,  0.88462238,  0.88131804,  1.70957306],
    [ 0.05003364, -0.40467741, -0.54535995, -1.54647732],
    [ 0.98236743, -1.10106763, -1.18504653, -0.2056499 ]]

    b1.data = [[ 1.48614836],
    [ 0.23671627],
    [-1.02378514]]

    Z1.data = [[-0.7129932 ,  0.62524497],
    [-0.16051336, -0.76883635],
    [-0.23003072,  0.74505627]]

    A2.data = [[ 1.97611078, -1.24412333],
    [-0.62641691, -0.80376609],
    [-2.41908317, -0.92379202]]

    W2.data = [[-1.02387576,  1.12397796, -0.13191423]]

    b2.data = [[-1.62328545]]

    Z2.data = [[ 0.64667545, -0.35627076]]

    let linear_cache_activation_1: [cacheObject, Val] = [{A: A1, W: W1, b: b1}, Z1];
    let linear_cache_activation_2: [cacheObject, Val] = [{A: A2, W: W2, b: b2}, Z2];
    let caches = [linear_cache_activation_1, linear_cache_activation_2]

    let grads = fn.LModelBackward(AL, Y, caches)
    console.log(grads)

    /*
    dA0 = [[ 0.          0.52257901]
    [ 0.         -0.3269206 ]
    [ 0.         -0.32070404]
    [ 0.         -0.74079187]]
    dA1 = [[ 0.12913162 -0.44014127]
    [-0.14175655  0.48317296]
    [ 0.01663708 -0.05670698]]
    dW1 = [[0.41010002 0.07807203 0.13798444 0.10502167]
    [0.         0.         0.         0.        ]
    [0.05283652 0.01005865 0.01777766 0.0135308 ]]
    dW2 = [[-0.39202432 -0.13325855 -0.04601089]]
    db1 = [[-0.22007063]
    [ 0.        ]
    [-0.02835349]]
    db2 = [[0.15187861]]
    */
}

function updateParameters_TEST() {
    let W1 = new Val([3,4])
    let b1 = new Val([3,1])
    let W2 = new Val([1,3])
    let b2 = new Val([1,1])

    let dW1 = new Val([3,4])
    let db1 = new Val([3,1])
    let dW2 = new Val([1,3])
    let db2 = new Val([1,1])

    W1.data = [[-0.41675785, -0.05626683, -2.1361961 ,  1.64027081],
    [-1.79343559, -0.84174737,  0.50288142, -1.24528809],
    [-1.05795222, -0.90900761,  0.55145404,  2.29220801]]

    b1.data = [[ 0.04153939],
    [-1.11792545],
    [ 0.53905832]]

    W2.data = [[-0.5961597 , -0.0191305 ,  1.17500122]]

    b2.data = [[-0.74787095]]

    dW1.data = [[ 1.78862847,  0.43650985,  0.09649747, -1.8634927 ],
    [-0.2773882 , -0.35475898, -0.08274148, -0.62700068],
    [-0.04381817, -0.47721803, -1.31386475,  0.88462238]]

    db1.data = [[0.88131804],
    [1.70957306],
    [0.05003364]]

    dW2.data = [[-0.40467741, -0.54535995, -1.54647732]]
    
    db2.data = [[0.98236743]]

    let parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    let grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    let new_params = fn.updateParameters(parameters, grads, 0.1)

    console.log(new_params)

}