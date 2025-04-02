import { assert } from "../src/js/utils/utils"
import { dot, broadcast } from "../src/js/Val/ops"
import { Val } from "../src/js/Val/val"

function broadcastTEST() {
    //TODO
    
    // both are val
    let v1: Val, v2: Val

    // case 1: v1:[x, 1] & v2:[x, y] 
    //  v1 will be [x, y]
    v1 = new Val([4, 1]); v2 = new Val([4, 3])
    broadcast(v1, v2)
    assert

    v1 = new Val([3, 1]); v2 = new Val([3, 4])
    assert
    broadcast(v1, v2)

    // case 2: v1:[x, y] & v2:[x, 1]
    // v2 will be [x, y]
    v1 = new Val([4, 3]); v2 = new Val([4, 1])
    assert
    broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // broadcast(v1, v2)

    // case 3: v1:[1, y] & v2:[x, y]
    // v1 will be [x, y]
    v1 = new Val([1, 4]); v2 = new Val([3, 4])
    assert
    broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // broadcast(v1, v2)

    // case 4: v1:[x, y] & v2:[1, y]
    // v2 will be [x, y]
    v1 = new Val([3, 4]); v2 = new Val([1, 4])
    assert
    broadcast(v1, v2)

    // v1 = new Val([3, 1]); v2 = new Val([3, 4])
    // assert
    // broadcast(v1, v2)

}

// dot
test("dot: (1, x) * (x, 1) => (1, 1)", () => {
    const a = new Val([1, 3])
    a.data = [[1,2,3]]
    const b = new Val([3, 1])
    b.data = [[4],[5],[6]]
    expect(dot(a, b).data).toEqual(Float64Array.from([32]));
    expect(dot(a, b).shape).toEqual([1, 1])
})

test("dot: (1, x) * (x, y) => (1, y)", () => {
    const a = new Val([1,2])
    a.data = [[1,2]]
    const b = new Val([2,3])
    b.data= [[1, -2, -1], [3, 0.5, -3.2]]
    expect(dot(a, b).data).toEqual(Float64Array.from([ 7, -1, -7.4 ]))
    expect(dot(a, b).shape).toEqual([1, 3])
})

test("dot: (x, y) * (y, 1) => (x, 1)", () => {
    const a = new Val([3,4])
    a.data = [[0.5,-3,-2,5], [6,-5,12,0.1], [0.6,-0.5,1,5]]
    const b = new Val([4,1])
    b.data= [[6],[-5],[-1.5],[-4]]
    expect(dot(a, b).data).toEqual(Float64Array.from([ 1, 42.6, -15.4 ]))
    expect(dot(a, b).shape).toEqual([3, 1])
})

test("dot: (x, y) * (y, z) => (x, z)", () => {
    const a = new Val([3,2])
    a.data = [[-4, 3], [2, -5], [1, 6]]
    const b = new Val([2,4])
    b.data= [[1, -0.4, 2, 5], [-3, 2, -1, 1]]
    expect(dot(a, b).data).toEqual(Float64Array.from([ -13, 7.6, -11, -17, 17, -10.8, 9, 5, -17, 11.6, -4, 11 ]))
    expect(dot(a, b).shape).toEqual([3, 4])
})

test("dot: 1D * 1D => scalar (dot product)", () => {
    const a = new Val([3])
    a.data = [1, 2, 3]
    const b = new Val([3])
    b.data = [4, 5, 6]
    expect(dot(a, b).data).toEqual(Float64Array.from([32]))
    expect(dot(a, b).shape).toEqual([1])
})

test("dot: (x) * (x, y) => (y)", () => {
    const a = new Val([3]) // this gets broadcasted into [1,3] and not [3,1] for the mat-mul (cause of how inner dims work)
    a.data = [1, 2, 3]
    const b = new Val([3, 2])
    b.data = [[4, 5], [6, 7], [8, 9]]
    expect(dot(a, b).data).toEqual(Float64Array.from([ 40, 46 ]))
    expect(dot(a, b).shape).toEqual([1, 2])
})

test("dot: (x, y) * (y) => (x)", () => {
    const a = new Val([2, 3])
    a.data = [[1, 2, 3], [4, 5, 6]]
    const b = new Val([3])
    b.data = [7, 8, 9]
    expect(dot(a, b).data).toEqual(Float64Array.from([ 50, 122 ]))
    expect(dot(a, b).shape).toEqual([2, 1])
})

test("dot: (1D * 2D) => (1, z)", () => {
    const a = new Val([3])
    a.data = [1, 2, 3]
    const b = new Val([3, 4])
    b.data = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    expect(dot(a, b).data).toEqual(Float64Array.from([56, 62, 68, 74]))
    expect(dot(a, b).shape).toEqual([1, 4])
})

test("dot: (2D * 1D) => (x)", () => {
    const a = new Val([3, 4])
    a.data = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    const b = new Val([4])
    b.data = [1, 2, 3, 4]
    expect(dot(a, b).data).toEqual(Float64Array.from([ 60, 100, 140 ]))
    expect(dot(a, b).shape).toEqual([3, 1])
})

test("dot: Edge case: zero matrix", () => {
    const a = new Val([2, 3])
    a.data = [[0, 0, 0], [0, 0, 0]]
    const b = new Val([3, 2])
    b.data = [[0, 0], [0, 0], [0, 0]]
    expect(dot(a, b).data).toEqual(Float64Array.from([0, 0, 0, 0]))
    expect(dot(a, b).shape).toEqual([2, 2])
})

test("dot: Mismatched dimensions (should raise an error)", () => {
    const a = new Val([2, 3])
    a.data = [[1, 2, 3], [4, 5, 6]]
    const b = new Val([4, 2])
    b.data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    expect(()=>dot(a, b)).toThrow();
})
