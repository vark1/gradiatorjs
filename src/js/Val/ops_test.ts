import { assert } from '../utils/utils';
import * as ops from '../Val/ops'
import { Val } from './val'

function broadcastTEST() {
    //TODO
    
    // both are val
    let v1: Val, v2: Val

    // case 1: v1:[x, 1] & v2:[x, y] 
    //  v1 will be [x, y]
    v1 = new Val([4, 1]); v2 = new Val([4, 3])
    ops.broadcast(v1, v2)
    assert

    v1 = new Val([3, 1]); v2 = new Val([3, 4])
    assert
    ops.broadcast(v1, v2)

    // case 2: v1:[x, y] & v2:[x, 1]
    // v2 will be [x, y]
    v1 = new Val([4, 3]); v2 = new Val([4, 1])
    assert
    ops.broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // ops.broadcast(v1, v2)

    // case 3: v1:[1, y] & v2:[x, y]
    // v1 will be [x, y]
    v1 = new Val([1, 4]); v2 = new Val([3, 4])
    assert
    ops.broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // ops.broadcast(v1, v2)

    // case 4: v1:[x, y] & v2:[1, y]
    // v2 will be [x, y]
    v1 = new Val([3, 4]); v2 = new Val([1, 4])
    assert
    ops.broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // ops.broadcast(v1, v2)

}