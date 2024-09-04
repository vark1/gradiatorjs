interface Value {
    data: number;
    label: string;
    grad: number;
    _prev: any[];
    _op: string;
    _backward: Function;
}

export const Value = function(data: number, label: string='', children: Value[]=[], _op: string=''){
    this.data = data
    this.label = label
    this.grad = 0
    this._prev = children
    this._op = _op
    this._backward = function(): void {}
}

Value.prototype = {
    toString: function() {
        return "Value(label=" + this.label + ", data="+ this.data +", grad= " + this.grad + ")"
    },
    add: function(other: any) {
        other = other instanceof Value ? other : new Value(other)
        let out = new Value(this.data + other.data, "", [this, other], '+')
        let _this = this    // Javascript :D
        function _backward() {
            _this.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        }
        out._backward = _backward

        return out
    },
    mul: function(other: any) {
        other = other instanceof Value ? other : new Value(other)
        let out = new Value(this.data * other.data, "", [this, other], '*')
        let _this = this
        function _backward() {
            _this.grad += other.data * out.grad
            other.grad += _this.data * out.grad
        }
        out._backward = _backward

        return out
    },
    pow: function(other: number) {
        let out = new Value(this.data**other, "", [this, ], "**" + other)
        let _this = this
        function _backward() { 
            _this.grad += other * this.data**(other-1) * out.grad
        }
        out._backward = _backward

        return out
    },
    div: function(other: any) {
        let out = this.mul(other.pow(-1))
        out._prev = [this, other]       // out's prev and op values will be self, other**-1 and *, this prevents that
        out._op = '/'
        return out
    },
    neg: function() {
        let out = this.mul(-1)
        out._op = '¬'
        out._prev = [this]
        return out
    },
    sub: function(other: any) {
        let out = this.add(other.neg())
        out._prev = [this, other]
        out._op = '-'
        return out
    },
    tanh: function() {
        let _this = this
        let x = this.data
        let t = (Math.exp(2*x) - 1)/(Math.exp(2*x) + 1)
        let out = new Value(t, "", [this], 'tanh')

        function _backward() { 
            _this.grad += (1-t**2) * out.grad 
        }
        out._backward = _backward

        return out
    },
    exp: function() {
        let _this = this
        let out = new Value(Math.exp(this.data), "", [this], 'exp')

        function _backward() {
            _this.grad += out.data * out.grad
        }
        out._backward = _backward

        return out
    },
    backward: function() {
        let topo = []
        let visited = new Set()
        function build_topo(v) {
            if (!visited.has(v)) {
                visited.add(v)
                for (let child of v._prev) {
                    build_topo(child)
                }
                topo.push(v)
            }
        }
        build_topo(this)
        this.grad = 1.0
        for (let node of topo.toReversed()) {
            node._backward()
        }
    }
}
