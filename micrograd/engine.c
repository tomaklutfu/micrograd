#include "engine.h"
#include <stdio.h>

// ValueStructDef(Type) defines Value struct in type TYPE
/*Usage
  ValueStructDef(Type)

  Type: float or double or any one word number type.
*/
// instantiateValue(Type) for instantiation of Value in type TYPE
/* Usage:
  Value(TYPE) *x = instantiateValue(float)(value, childs, nchild, backward);

  x:      Pointer to Value(TYPE) (Value in type TYPE)
  value:  A value in TYPE
  childs: Pointer to an array of pointer(s) to Value(TYPE)
  nchild: Number of elements in childs
  backward: A function used to make backward gradient propagation.
  x.data stores value.
  x.grad is initialized with 0.
  x.parentreference is internally set to 0 for correctly ordered backpropagation.
*/

// addValue(Type) for addition function of Values in type TYPE
/* Usage:
  Value(TYPE) *res = addValue(float)(childs, nchild);

  res:    Pointer to Value(Type)
  childs: Pointer to an array of pointer(s) to Value(TYPE)
  nchild: Number of elements in childs
  res.data stores the addition of the data fields in all the elements in childs.
  res.grad is initialized with 0.
  res.backward is internally assigned from this function.
  res.parentreference is internally set to 0 for correctly ordered backpropagation.
  All the elements in childs have their parentreference field incremented.
*/
// mulValue(Type) for multiplication function of Values in type TYPE
/* Usage:
  Value(TYPE) *res = mulValue(float)(childs, nchild);

  res:    Pointer to Value(Type)
  childs: Pointer to an array of pointer(s) to Value(TYPE
  nchild: Number of elements in childs array
  res.data stores the product of the data fields in all the elements in childs.
  res.grad is initialized with 0.
  res.backward is internally assigned from this function.
  res.parentreference is internally set to 0 for correctly ordered backpropagation.
  All the elements in childs have their parentreference field incremented.
*/
// backward() for backpropagation of gradient
/* Usage:
  v->grad = (TYPE)1; //First initialize output gradient to 1.
  v->backward();
  v: Pointer to Value(Type)
*/

int main(){
  Value(float) *x = instantiateValue(float)(5.0f, NULL, 0, NULL); // Leaf value not backwarding gradients
  Value(float) *t = instantiateValue(float)(0.0f, NULL, 0, NULL); // Leaf value not backwarding gradients
  Value(float) *xs[2] = {x, x};
  Value(float) *y = addValue(float)(xs, 2); // x+x
  Value(float) *z = mulValue(float)(xs, 2); // x*x
  Value(float) *x2y[3] = {x, y, x};
  Value(float) *x2z[3] = {x, z, x};
  Value(float) *xyz[3] = {x, y, z};
  Value(float) *xyt[3] = {x, y, t};
  // (x * y * x) * (x * z * x) * (x + y + z) * (x * y * t)=(2x³)(x⁴)(3x+x²)(2x²t)
  Value(float) *ochilds[4] = {mulValue(float)(x2y, 3), mulValue(float)(x2z, 3), addValue(float)(xyz, 3), mulValue(float)(xyt, 3)};
  Value(float) *o = mulValue(float)(ochilds, 4);

  puts("How many parents references each Value?");
  printf("o.parentreference=%d\n", o->parentreference);
  printf("o.childs[0].parentreference=%d\n", o->childs[0]->parentreference);
  printf("o.childs[1].parentreference=%d\n", o->childs[1]->parentreference);
  printf("o.childs[2].parentreference=%d\n", o->childs[2]->parentreference);
  printf("o.childs[3].parentreference=%d\n", o->childs[2]->parentreference);
  printf("z.parentreference=%d\n", z->parentreference);
  printf("y.parentreference=%d\n", y->parentreference);
  printf("t.parentreference=%d\n", t->parentreference);
  printf("x.parentreference=%d\n", x->parentreference);

  puts("");
  puts("Set output gradient to 1");
  o->grad = 1;
  puts("Backpropagate gradients from output");
  o->backward(o);

  puts("");
  puts("See the resutls in each Value");
  printf("o.data=%lf o.grad=%lf\n", o->data, o->grad);
  printf("o.childs[0].data=%lf o.childs[0].grad=%lf\n", o->childs[0]->data, o->childs[0]->grad);
  printf("o.childs[1].data=%lf o.childs[1].grad=%lf\n", o->childs[1]->data, o->childs[1]->grad);
  printf("o.childs[2].data=%lf o.childs[2].grad=%lf\n", o->childs[2]->data, o->childs[2]->grad);
  printf("o.childs[3].data=%lf o.childs[3].grad=%lf\n", o->childs[3]->data, o->childs[3]->grad);
  printf("z.data=%lf z.grad=%lf\n", z->data, z->grad);
  printf("y.data=%lf y.grad=%lf\n", y->data, y->grad);
  printf("t.data=%lf t.grad=%lf\n", t->data, t->grad);
  printf("x.data=%lf x.grad=%lf\n", x->data, x->grad);

  puts("");
  puts("Check if parents releases references in backpropagation!");
  printf("o.parentreference=%d\n", o->parentreference);
  printf("o.childs[0].parentreference=%d\n", o->childs[0]->parentreference);
  printf("o.childs[1].parentreference=%d\n", o->childs[1]->parentreference);
  printf("o.childs[2].parentreference=%d\n", o->childs[2]->parentreference);
  printf("o.childs[3].parentreference=%d\n", o->childs[2]->parentreference);
  printf("z.parentreference=%d\n", z->parentreference);
  printf("y.parentreference=%d\n", y->parentreference);
  printf("t.parentreference=%d\n", t->parentreference);
  printf("x.parentreference=%d\n", x->parentreference); 

  return 0;
}
