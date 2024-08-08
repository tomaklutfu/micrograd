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

ValueStructDef(float)  // Define Value type for float
ValueStructDef(double) // Define Value type for double