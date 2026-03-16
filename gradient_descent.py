#!/usr/bin/env python3
"""Gradient descent optimizers — SGD, Momentum, Adam."""
import math

class SGD:
    def __init__(self, lr=0.01): self.lr = lr
    def step(self, params, grads): return [p - self.lr*g for p,g in zip(params,grads)]

class Momentum:
    def __init__(self, lr=0.01, beta=0.9): self.lr=lr; self.beta=beta; self.v=None
    def step(self, params, grads):
        if self.v is None: self.v = [0]*len(params)
        self.v = [self.beta*v + self.lr*g for v,g in zip(self.v,grads)]
        return [p - v for p,v in zip(params,self.v)]

class Adam:
    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr;self.b1=b1;self.b2=b2;self.eps=eps;self.m=None;self.v=None;self.t=0
    def step(self, params, grads):
        self.t += 1
        if self.m is None: self.m=[0]*len(params); self.v=[0]*len(params)
        self.m = [self.b1*m+(1-self.b1)*g for m,g in zip(self.m,grads)]
        self.v = [self.b2*v+(1-self.b2)*g*g for v,g in zip(self.v,grads)]
        mh = [m/(1-self.b1**self.t) for m in self.m]
        vh = [v/(1-self.b2**self.t) for v in self.v]
        return [p - self.lr*m/(math.sqrt(v)+self.eps) for p,m,v in zip(params,mh,vh)]

def minimize(f, grad_f, x0, optimizer, steps=1000):
    x = list(x0)
    for _ in range(steps): x = optimizer.step(x, grad_f(x))
    return x

def main():
    f = lambda x: x[0]**2 + x[1]**2
    gf = lambda x: [2*x[0], 2*x[1]]
    for name, opt in [("SGD",SGD(0.1)),("Momentum",Momentum(0.1)),("Adam",Adam(0.1))]:
        r = minimize(f, gf, [5.0, 3.0], opt, 100)
        print(f"{name}: x={[round(v,6) for v in r]}, f={f(r):.8f}")

if __name__ == "__main__": main()
