#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>

// TODO: May need asserts for childs[i] != NULL
// TODO: Add more functions to backproping

#define ValueStructDef(Type)     \
  struct Value##Type             \
  {                              \
    Type data;                   \
    Type grad;                   \
    struct Value##Type **childs; \
    int nchild;                  \
    int parentreference;         \
    void (*backward)(struct Value##Type *self); \
  }; \
  typedef struct Value##Type Value##Type; \
  typedef void (*backward##Type)(Value##Type *self);\
  Value##Type *instantiateValue##Type(Type data, Value##Type **childs, int n, backward##Type backward)\
  {\
    Value##Type *self = (Value##Type *)malloc(sizeof(Value##Type));\
    self->data = data; \
    self->grad = 0*data; \
    self->childs = childs; \
    self->nchild = n; \
    self->parentreference = 0;  /*For backprop topological order.*/\
    self->backward = backward; \
    return self; \
  } \
  void backwardChildValue##Type(Value##Type **childs, int n)\
  { \
    for(int i = 0; i < n; i++) \
    {\
      childs[i]->parentreference--; /*One parent is done backwarding.*/\
      if(childs[i]->backward != NULL && childs[i]->parentreference <= 0) /*Only backwardable if all the parents are done backwarding!*/ \
        childs[i]->backward(childs[i]); \
    }\
  } \
  void add_backward##Type(Value##Type *self) \
  { \
    Value##Type **childs = self->childs; \
    int n = self->nchild; \
    assert(childs != 0 && n >= 2); \
    for(int i = 0; i < n; i++) \
      childs[i]->grad += self->grad; \
    backwardChildValue##Type(childs, n); \
  } \
  Value##Type *addValue##Type(Value##Type **childs, int n) \
  { \
    assert(childs != 0 && n >= 2); \
    Type sum = (Type)0; \
    for(int i = 0; i < n; i++) \
    {\
      sum += childs[i]->data; \
      childs[i]->parentreference++; \
    }\
    return instantiateValue##Type(sum, childs, n, add_backward##Type);\
  } \
/*void pow_backward##Type(Value##Type *self) \
  { \
    Value##Type **childs = self->childs; \
    int n = self->nchild; \
    assert(childs != 0 && n == 2); \
    childs[0]->grad += childs[0]->data == 0 ? self->data : self->grad*childs[1]->data*self->data/childs[0]->data; \
    childs[1]->grad += self->grad*self->data*log(childs[0]->data); \
    backwardChildValue##Type(childs, n); \
  } \
  Value##Type *powValue##Type(Value##Type **childs, int n) \
  { \
    assert(childs != 0 && n == 2); \
    Type res = pow(childs[0]->data, childs[1]->data); \
    childs[0]->parentreference++; \
    childs[1]->parentreference++; \
    return instantiateValue##Type(res, childs, n, pow_backward##Type);\
  } */\
  Type prodotherchilds##Type(Value##Type **childs, int n, int i)\
  {\
    Type prod = (Type)1;\
    for(int j = 0; j < n; j++)\
      prod *= (i==j ? (Type)1 : childs[j]->data);\
    return prod;\
  } \
  void mul_backward##Type(Value##Type *self) \
  { \
    Value##Type **childs = self->childs; \
    int n = self->nchild; \
    assert(childs != 0 && n >= 2); \
    for(int i = 0; i < n; i++) \
      childs[i]->grad += self->data !=0 ? self->grad*self->data/childs[i]->data : (childs[i]->data != 0 ? self->data : self->grad*prodotherchilds##Type(childs, n, i)); \
    backwardChildValue##Type(childs, n); \
  } \
  Value##Type *mulValue##Type(Value##Type **childs, int n) \
  { \
    assert(childs != 0 && n >= 2); \
    Type prod = (Type)1; \
    for(int i = 0; i < n; i++) \
    {\
      prod *= childs[i]->data; \
      childs[i]->parentreference++;\
    }\
    return instantiateValue##Type(prod, childs, n, mul_backward##Type);\
  } \

#define function(Fun, Type) Fun##Value##Type
#define bfunction(Fun, Type) Fun##_backward##Type

#define Value(Type) Value##Type
#define instantiateValue(Type) instantiateValue##Type
#define addValue(Type) addValue##Type
#define mulValue(Type) mulValue##Type

#ifdef DefMain
ValueStructDef(float)
ValueStructDef(double)
int main(){
  Value(float) *x = instantiateValue(float)(5.0f, NULL, 0, NULL); // Leaf value not backwarding gradients
  Value(float) *t = instantiateValue(float)(0.0f, NULL, 0, NULL); // Leaf value not backwarding gradients
  Value(float) *xs[2] = {x, x};
  Value(float) *y = addValue(float)(xs, 2); // 2x
  Value(float) *z = mulValue(float)(xs, 2); // x²
  Value(float) *x2y[3] = {x, y, x};
  Value(float) *x2z[3] = {x, z, x};
  Value(float) *xyz[3] = {x, y, z};
  Value(float) *xyt[3] = {x, y, t};
  // (x²y)*(x²z)*(x + y + z)=(2x²)(x⁴)*(3x+x²)(x*y*t)
  Value(float) *ochilds[4] = {mulValue(float)(x2y, 3), mulValue(float)(x2z, 3), addValue(float)(xyz, 3), mulValue(float)(xyt, 3)};
  Value(float) *o = mulValue(float)(ochilds, 4);

  printf("o.parentreference=%d\n", o->parentreference);
  printf("o.childs[0].parentreference=%d\n", o->childs[0]->parentreference);
  printf("o.childs[1].parentreference=%d\n", o->childs[1]->parentreference);
  printf("o.childs[2].parentreference=%d\n", o->childs[2]->parentreference);
  printf("o.childs[3].parentreference=%d\n", o->childs[2]->parentreference);
  printf("z.parentreference=%d\n", z->parentreference);
  printf("y.parentreference=%d\n", y->parentreference);
  printf("t.parentreference=%d\n", t->parentreference);
  printf("x.parentreference=%d\n", x->parentreference);
  o->grad = 1;
  o->backward(o);
  printf("o.data=%lf o.grad=%lf\n", o->data, o->grad);
  printf("o.childs[0].data=%lf o.childs[0].grad=%lf\n", o->childs[0]->data, o->childs[0]->grad);
  printf("o.childs[1].data=%lf o.childs[1].grad=%lf\n", o->childs[1]->data, o->childs[1]->grad);
  printf("o.childs[2].data=%lf o.childs[2].grad=%lf\n", o->childs[2]->data, o->childs[2]->grad);
  printf("o.childs[3].data=%lf o.childs[3].grad=%lf\n", o->childs[3]->data, o->childs[3]->grad);
  printf("z.data=%lf z.grad=%lf\n", z->data, z->grad);
  printf("y.data=%lf y.grad=%lf\n", y->data, y->grad);
  printf("t.data=%lf t.grad=%lf\n", t->data, t->grad);
  printf("x.data=%lf x.grad=%lf\n", x->data, x->grad);
  return 0;
}
#endif