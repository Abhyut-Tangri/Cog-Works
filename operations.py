class Operation(object):
    """ All `Operations` represent two-variable mathematical functions, e.g. +,-,*,/,** 
        __call__ accepts two `Number` objects, and returns a Python-numeric result (int, float)
    """
    def partial_a(self):
        """ Computes the partial derivative of this operation with respect to a: d(op)/da"""
        raise NotImplementedError

    def partial_b(self):
        """ Computes the partial derivative of this operation with respect to b: d(op)/db"""
        raise NotImplementedError

    def __call__(self, a, b):
        """ Computes the forward pass of this operation: op(a, b) -> output"""
        raise NotImplementedError

    def backprop(self, grad):
        """ Calls .backprop for self.a and self.b, passing to it dF/da and
            dF/db, respectively, where F represents the terminal `Number`-instance node 
            in the computational graph, which originally invoked `F.backprop().
             
            (In short: F is the variable with respect to which all of the partial derivatives 
            are being computed, for this back propagation)
            
            Parameters
            ----------
            grad : Union[int, float]
                dF/d(op) - The partial derivative of F with respect to the output of this operation.
            """
        self.a.backprop(self.partial_a() * grad)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad)


    def null_gradients(self):
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()


class Add(Operation):
    def __repr__(self): return "+"

    def __call__(self, a, b):
        """ Adds two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data + b.data
    
    def partial_a(self):
        """ Returns d(a + b)/da """
        return 1
    
    def partial_b(self):
        """ Returns d(a + b)/db """
        return 1
    def backprop(self, grad):
        # These call backprop on Number objects, which have .grad
        self.a.backprop(self.partial_a() * grad)
        self.b.backprop(self.partial_b() * grad)

class Multiply(Operation):
    def __repr__(self): return "*"

    def __call__(self, a, b):
        """ Nultiplies two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data * b.data
    
    def partial_a(self):
        """ Returns d(a * b)/da as int or float"""
        return self.b.data
    
    def partial_b(self):
        """ Returns d(a * b)/db as int or float"""
        return self.a.data
    def backprop(self, grad):
        # These call backprop on Number objects, which have .grad
        self.a.backprop(self.partial_a() * grad)
        self.b.backprop(self.partial_b() * grad)

class Subtract(Operation):
    def __repr__(self): return "-"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data - b.data

    def partial_a(self):
        return 1

    def partial_b(self):
        return -1
    def backprop(self, grad):
        # These call backprop on Number objects, which have .grad
        self.a.backprop(self.partial_a() * grad)
        self.b.backprop(self.partial_b() * grad)


class Divide(Operation):
    def __repr__(self): return "/"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data / b.data

    def partial_a(self):
        return 1 / self.b.data

    def partial_b(self):
        return -self.a.data / (self.b.data ** 2)
    def backprop(self, grad):
        # These call backprop on Number objects, which have .grad
        self.a.backprop(self.partial_a() * grad)
        self.b.backprop(self.partial_b() * grad)


class Power(Operation):
    def __repr__(self): return "**"

    def __call__(self, a, b):
        self.a = a
        self.b = b
        return a.data ** b.data

    def partial_a(self):
        return self.b.data * (self.a.data ** (self.b.data - 1))

    def partial_b(self):
        import math
        return (self.a.data ** self.b.data) * math.log(self.a.data)
    def backprop(self, grad):
        # These call backprop on Number objects, which have .grad
        self.a.backprop(self.partial_a() * grad)
        self.b.backprop(self.partial_b() * grad)