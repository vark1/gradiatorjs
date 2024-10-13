import { assert } from "../src/js/utils/utils"
import { dot } from "../src/js/Val/ops"
import { Val } from "../src/js/Val/val"

function dot_test() {
    // (1, x) * (x, 1) => (1, 1)
    let a = new Val([1, 3])
    a.data = [[1,2,3]]
    let b = new Val([3, 1])
    b.data = [[4],[5],[6]]
    console.log(dot(a, b))  //[32]

    // (1, x) * (x, y) => (1, y)
    let r = new Val([1,2])
    r.data = [[1,2]]
    let s = new Val([2,3])
    s.data= [[1, -2, -1], [3, 0.5, -3.2]]
    console.log(dot(r,s))   //[ 7, -1, -7.4 ]


    // (x, y) * (y, 1) => (x, 1)
    let p = new Val([3,4])
    p.data = [[0.5,-3,-2,5], [6,-5,12,0.1], [0.6,-0.5,1,5]]
    let q = new Val([4,1])
    q.data= [[6],[-5],[-1.5],[-4]]
    console.log(dot(p,q))   //[ 1, 42.6, -15.4 ]

    // (x, y) * (y, z) => (x, z)
    let i = new Val([3,2])
    i.data = [[-4, 3], [2, -5], [1, 6]]
    let j = new Val([2,4])
    j.data= [[1, -0.4, 2, 5], [-3, 2, -1, 1]]
    console.log(dot(i, j))  //[ -13, 7.6, -11, -17, 17, -10.8, 9, 5, -17, 11.6, -4, 11 ]

    // 1D * 1D => scalar (dot product)
    let u = new Val([3])
    u.data = [1, 2, 3]
    let v = new Val([3])
    v.data = [4, 5, 6]
    console.log(dot(u, v))  // [32]

    // (x) * (x, y) => (y)
    let vec1 = new Val([3])
    vec1.data = [1, 2, 3]
    let mat1 = new Val([3, 2])
    mat1.data = [[4, 5], [6, 7], [8, 9]]
    console.log(dot(vec1, mat1))  //[ 40, 46 ]

    // (x, y) * (y) => (x)
    let mat2 = new Val([2, 3])
    mat2.data = [[1, 2, 3], [4, 5, 6]]
    let vec2 = new Val([3])
    vec2.data = [7, 8, 9]
    console.log(dot(mat2, vec2))  //[ 50, 122 ]

    // (1D * 2D) => (1, z)
    let vec3 = new Val([3])
    vec3.data = [1, 2, 3]
    let mat3 = new Val([3, 4])
    mat3.data = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    console.log(dot(vec3, mat3))  //[ [56, 62, 68, 74] ]

    // (2D * 1D) => (x)
    let mat4 = new Val([3, 4])
    mat4.data = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    let vec4 = new Val([4])
    vec4.data = [1, 2, 3, 4]
    console.log(dot(mat4, vec4))  //[ 60, 140, 220 ]

    // Edge case: zero matrix
    let zero1 = new Val([2, 3])
    zero1.data = [[0, 0, 0], [0, 0, 0]]
    let zero2 = new Val([3, 2])
    zero2.data = [[0, 0], [0, 0], [0, 0]]
    console.log(dot(zero1, zero2))  //[ [0, 0], [0, 0] ]

    // Edge case: empty matrix (should raise an error or handle appropriately)
    try {
        let empty1 = new Val([0, 2])
        let empty2 = new Val([2, 0])
        console.log(dot(empty1, empty2))  // Should handle gracefully
    } catch (e) {
        console.log("Error: Empty matrix case caught", e);
    }

    // Mismatched dimensions (should raise an error)
    try {
        let badDim1 = new Val([2, 3])
        badDim1.data = [[1, 2, 3], [4, 5, 6]]
        let badDim2 = new Val([4, 2])
        badDim2.data = [[1, 2], [3, 4], [5, 6], [7, 8]]
        console.log(dot(badDim1, badDim2))  // Should raise an assertion error
    } catch (e) {
        console.log("Error: Dimension mismatch case caught", e);
    }
}

dot_test()