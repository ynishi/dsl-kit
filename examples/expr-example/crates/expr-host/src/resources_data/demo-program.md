# expr DSL — demo program

The default program `ExprHost` loads when it starts up. A single
`Let` binding wrapping a `Mul` of `Add x y` and `Var z`:

```text
let x = 3 in (x + y) * z
```

With `y = 5` and `z = 2` (`ExprHost`'s canned defaults), this
evaluates to `(3 + 5) * 2 = 16`.

## Source

```rust,ignore
pub fn demo_program(ids: &IdGen) -> Expr {
    let x_lit = Expr::Lit { id: ids.node(), value: 3 };
    let x_ref = Expr::Var { id: ids.node(), name: "x".into() };
    let y_ref = Expr::Var { id: ids.node(), name: "y".into() };
    let z_ref = Expr::Var { id: ids.node(), name: "z".into() };
    let add = Expr::Add { id: ids.node(), lhs: Box::new(x_ref), rhs: Box::new(y_ref) };
    let mul = Expr::Mul { id: ids.node(), lhs: Box::new(add), rhs: Box::new(z_ref) };
    Expr::Let {
        id: ids.node(),
        name: "x".into(),
        value: Box::new(x_lit),
        body: Box::new(mul),
    }
}
```

## Structure

```text
Let "x"
├── Lit 3
└── Mul
    ├── Add
    │   ├── Var "x"
    │   └── Var "y"
    └── Var "z"
```

`dsl_kit_ast` returns the same shape with real `NodeId`s and full
`Expr::summary()` labels.

## Driving it end-to-end

The `Let` binds `x` internally, so `Var "x"` resolves without a host
round-trip. The two suspensions come from `y` and `z`:

```text
1. dsl_kit_step { mode: "to_yield" }   # suspends at Var "y"
2. dsl_kit_resolve { result: "5" }      # bind y = 5
3. dsl_kit_step { mode: "to_yield" }   # suspends at Var "z"
4. dsl_kit_resolve { result: "2" }      # bind z = 2
5. dsl_kit_step { mode: "to_yield" }   # Done with value 16
```

Or short-circuit the whole run:

```text
dsl_kit_step { mode: "to_done" }        # host supplies canned defaults for y, z
```

After `to_done` the final value (16) and the two resolved bindings
appear under `dsl_kit_state.results`.
