
import { Val } from '../src/js/Val/val'
import { gaussianRandom } from '../src/js/utils/utils_num';
import * as op from '../src/js/Val/ops'

test("Create scalar Val", () => {
  const a = new Val([], 5); // Scalar
  expect(a.size).toBe(1);
  expect(a.data).toEqual(Float64Array.from([5]));
});

test("Transpose 2D matrix", () => {
  const a = new Val([2, 3], 0);
  a.data = Float64Array.from([0, 1, 2, 3, 4, 5]);
  const b = a.T;
  expect(b.shape).toEqual([3, 2]);
  expect(b.data).toEqual(Float64Array.from([0, 3, 1, 4, 2, 5]));
});

test("Transpose backward pass", () => {
  const a = new Val([2, 2], 1);
  const b = a.T;
  b.grad = Float64Array.from([1, 2, 3, 4]);
  b.backward();
  expect(a.grad).toEqual(Float64Array.from([1, 3, 2, 4])); // Should pass after fix
});

test("Clone Val", () => {
  const a = new Val([2, 2], 5);
  const b = a.clone();
  expect(b.data).toEqual(a.data);
  expect(b.shape).toEqual(a.shape);
});

test("Backward on vector", () => {
  const a = new Val([3], 2);
  a.backward();
  expect(a.grad).toEqual(new Float64Array([0, 0, 0])); // Should pass after fix
});

test("Reshape Val", () => {
  const a = new Val([2, 3], 0);
  const b = a.reshape([3, 2]);
  expect(b.shape).toEqual([3, 2]);
});

test("Set invalid data shape", () => {
  const a = new Val([2, 2], 0);
  expect(() => {
    a.data = [1, 2, 3]; // Incorrect size (3 vs 4)
  }).toThrow();
});

describe('Val (Value Class)', () => {
  describe('Core Functionality', () => {
    test('creates scalar with correct properties', () => {
      const v = new Val([], 5);
      expect(v.dim).toBe(0);
      expect(v.size).toBe(1);
      expect(v.data).toEqual(new Float64Array([5]));
      expect(v.shape).toEqual([]);
    });

    test('creates tensor with correct shape/size', () => {
      const v = new Val([2, 3], 1);
      expect(v.dim).toBe(2);
      expect(v.size).toBe(6);
      expect(v.shape).toEqual([2, 3]);
      expect(v.data).toEqual(new Float64Array([1, 1, 1, 1, 1, 1]));
    });

    test('data assignment validates shape', () => {
      const v = new Val([2, 2]);
      v.data = [[1, 2], [3, 4]];
      expect(v.data).toEqual(new Float64Array([1, 2, 3, 4]));

      expect(() => {
        v.data = [1, 2, 3]; // Wrong shape
      }).toThrow();
    });
  });

  describe('Operations', () => {
    let a: Val, b: Val;

    beforeEach(() => {
      a = new Val([2]);
      a.data = [1, 2]
      b = new Val([2]);
      b.data = [3, 4]
    });

    test('clone creates independent copy', () => {
      const c = a.clone();
      expect(c.data).toEqual(a.data);
      c.data[0] = 5;
      expect(a.data[0]).toBe(1); // Original unchanged
    });

    test('transpose 2D matrix', () => {
      const mat = new Val([2, 3], 0);
      mat.data = [[1, 2, 3], [4, 5, 6]];
      const transposed = mat.T;
      
      expect(transposed.shape).toEqual([3, 2]);
      expect(transposed.data).toEqual(new Float64Array([1, 4, 2, 5, 3, 6]));
    });

    test('transpose gradient propagation', () => {
      const mat = new Val([2, 2], 0, true);
      mat.data = [[1, 2], [3, 4]];
      const transposed = mat.T;
      
      transposed.grad = new Float64Array([1, 2, 3, 4]);
      transposed.backward();
      
      expect(mat.grad).toEqual(new Float64Array([1, 3, 2, 4]));
    });
  });

  describe('Autograd System', () => {
    test('backward propagates through computation graph', () => {
      const a = new Val([], 2, true);
      const b = new Val([], 3, true);
      const c = new Val([], 4, true);
      const d = op.add(op.mul(a, b), c)  // d = a*b + c

      d.backward();
      
      expect(a.grad[0]).toBeCloseTo(3); // ∂d/∂a = b = 3
      expect(b.grad[0]).toBeCloseTo(2); // ∂d/∂b = a = 2
      expect(c.grad[0]).toBeCloseTo(1); // ∂d/∂c = 1
    });

    test('topological ordering handles complex graphs', () => {
      const x = new Val([], 2, true);
      const y = op.mul(op.mul(x, x), x)// y = x³
      
      y.backward();
      expect(x.grad[0]).toBeCloseTo(12); // dy/dx = 3x² = 12
    });

    test('requiresGrad controls gradient computation', () => {
      const a = new Val([], 1, false); // No grad
      const b = new Val([], 2, true);
      const c = op.add(a, b);
      
      c.backward();
      expect(a.grad[0]).toBe(0); // Not tracked
      expect(b.grad[0]).toBe(1); // Tracked
    });
  });

  describe('Utility Methods', () => {
    test('gradVal returns gradient as Val', () => {
      const v = new Val([2]);
      v.data = [1, 2]
      v.grad = new Float64Array([3, 4]);
      const gradVal = v.gradVal();
      
      expect(gradVal.data).toEqual(new Float64Array([3, 4]));
      expect(gradVal.requiresGrad).toBe(false);
    });

    test('randn initializes with normal distribution', () => {
      jest.spyOn(Math, 'random').mockReturnValue(0.5);
      const v = new Val([3]).randn();
      
      // Mocked gaussianRandom with fixed seed
      expect(v.data.length).toBe(3);
      expect(v.data[0]).toBeCloseTo(0, 1); // Roughly normal
    });
  });

  describe('Edge Cases', () => {
    test('empty tensor creation', () => {
      const v = new Val([0]);
      expect(v.size).toBe(0);
      expect(v.data).toEqual(new Float64Array([]));
    });

    test('backward on non-scalar throws if not initialized', () => {
      const v = new Val([2]);
      v.data = [1, 2]
      expect(() => v.backward()).toThrow(); // Gradients not initialized
    });
  });
});
