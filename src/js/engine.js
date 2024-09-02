var Value = function(data, children=[], _op='', label=''){
    this.data = data
    this.grad = 0
    this._prev = children
    this._op = _op
    this.label = label
}

Value.prototype = {
    toString: function() {
        return "Value(data={"+ this.data +"})"
    },
    add: function(other) {
         other = other instanceof Value ? other : new Value(other)
         out = new Value(this.data + other.data, [this, other], '+')
         return out
    },
    mul: function(other) {
        other = other instanceof Value ? other : new Value(other)
        out = new Value(this.data * other.data, [this, other], '*')
        return out
    },
    pow: function(other) {
        try{
            if(!(typeof other === 'number')){
                throw new Error("Passed value is not a number");
            }
        } catch(err){
            console.log("An error occurred: ", err.message);
            return
        }
        out = new Value(this.data**other, [this, ], "**" + other)
        return out
    },
    div: function(other) {
        out = this.mul(other.pow(-1))
        out._prev = [this, other]       // out's prev and op values will be self, other**-1 and *, this prevents that
        out._op = '/'
        return out
    },
    neg: function() {
        out = this.mul(-1)
        out._op = '¬'
        out._prev = [self]
        return out
    },
    sub: function(other) {
        out = this.add(other.neg())
        out._prev = [this, other]
        out._op = '-'
        return out
    },
    tanh: function() {
        x = this.data
        t = (Math.exp(2*x) - 1)/(Math.exp(2*x) + 1)
        out = new Value(t, [this], 'tanh')
        return out
    },
    exp: function() {
        x = this.data
        out = new Value(Math.exp(x), [self], 'exp')
        return out
    }
    
}

a = new Value (2.0, [], "", 'a')
b = new Value (-3.0, [], "", 'b')
c = new Value (10, label='c')
L = b.div(a)
L.label = 'L'
console.log(L)