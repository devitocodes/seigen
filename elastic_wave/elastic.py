#!/usr/bin/env python

from firedrake import *

# Components of velocity equation RHS
def fx(w, s0, n, absorption=None):
   fx = -inner(grad(w[0])[0], s0[0])*dx + inner(avg(s0[0]), jump(w[0], n[0]))*dS - inner(grad(w[0])[1], s0[1])*dx + inner(avg(s0[1]), jump(w[0], n[1]))*dS
   if(absorption):
      fx += -inner(w[0], absorption*u[0])*dx
   return fx
   
def fy(w, s0, n, absorption=None):
   fy = -inner(grad(w[1])[0], s0[2])*dx + inner(avg(s0[2]), jump(w[1], n[0]))*dS - inner(grad(w[1])[1], s0[3])*dx + inner(avg(s0[3]), jump(w[1], n[1]))*dS - inner(w[1], absorption*u[1])*dx
   if(absorption):
      fy += -inner(w[1], absorption*u[1])*dx
   return fy

# Components of stress equation RHS
def gxx(v, u1, n, l, mu, source=None):
   gxx =  - (l + 2*mu)*inner(grad(v[0])[0], u1[0])*dx \
          + (l + 2*mu)*inner(jump(v[0], n[0]), avg(u1[0]))*dS \
          + (l + 2*mu)*inner(v[0], u1[0]*n[0])*ds \
          - l*inner(grad(v[0])[1], u1[1])*dx \
          + l*inner(jump(v[0], n[1]), avg(u1[1]))*dS \
          + l*inner(v[0], u1[1]*n[1])*ds
   if(source):
      gxx += inner(v[0], source)*dx
   return gxx
   
def gxy(v, u1, n, l, mu):
   return - mu*(inner(grad(v[1])[0], u1[1]))*dx \
          + mu*(inner(v[1], u1[1]*n[0]))*ds \
          + mu*(inner(jump(v[1], n[0]), avg(u1[1])))*dS \
          - mu*(inner(grad(v[1])[1], u1[0]))*dx \
          + mu*(inner(v[1], u1[0]*n[1]))*ds \
          + mu*(inner(jump(v[1], n[1]), avg(u1[0])))*dS

def gyx(v, u1, n, l, mu):
   return - mu*(inner(grad(v[2])[0], u1[1]))*dx \
          + mu*(inner(v[2], u1[1]*n[0]))*ds \
          + mu*(inner(jump(v[2], n[0]), avg(u1[1])))*dS \
          - mu*(inner(grad(v[2])[1], u1[0]))*dx \
          + mu*(inner(v[2], u1[0]*n[1]))*ds \
          + mu*(inner(jump(v[2], n[1]), avg(u1[0])))*dS
          
def gyy(v, u1, n, l, mu, source=None):    
   gyy =  - l*inner(grad(v[3])[0], u1[0])*dx \
          + l*inner(v[3], u1[0]*n[0])*ds \
          + l*inner(jump(v[3], n[0]), avg(u1[0]))*dS \
          - (l + 2*mu)*inner(grad(v[3])[1], u1[1])*dx \
          + (l + 2*mu)*inner(jump(v[3], n[1]), avg(u1[1]))*dS \
          + (l + 2*mu)*inner(v[3], u1[1]*n[1])*ds
   if(source):
      gyy += inner(v[3], source)*dx
   return gyy
   
