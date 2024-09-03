interface Value {
    data: number;
    label: string;
    grad: number;
    children: any[];
    _op: string;
}

export const Value = function(data: number, label: string='', children: Value[]=[], _op: string=''){
    this.data = data
    this.label = label
    this.grad = 0
    this._prev = children
    this._op = _op
}

Value.prototype = {
    toString: function() {
        return "Value(data={"+ this.data +"})"
    },
    add: function(other: any) {
         other = other instanceof Value ? other : new Value(other)
         let out = new Value(this.data + other.data, "", [this, other], '+')
         return out
    },
    mul: function(other: any) {
        other = other instanceof Value ? other : new Value(other)
        let out = new Value(this.data * other.data, "", [this, other], '*')
        return out
    },
    pow: function(other: number) {
        let out = new Value(this.data**other, "", [this, ], "**" + other)
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
        let x = this.data
        let t = (Math.exp(2*x) - 1)/(Math.exp(2*x) + 1)
        let out = new Value(t, "", [this], 'tanh')
        return out
    },
    exp: function() {
        let out = new Value(Math.exp(this.data), "", [this], 'exp')
        return out
    }
}
