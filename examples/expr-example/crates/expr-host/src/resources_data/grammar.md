# expr DSL — grammar

The `expr` DSL is the second reference DSL shipped with `dsl-kit`.
Its AST is one enum with six variants covering arithmetic, bindings,
and control flow.

```text
Expr ::= Lit { id, value: i64 }
       | Var { id, name: String }
       | Add { id, lhs: Box<Expr>, rhs: Box<Expr> }
       | Mul { id, lhs: Box<Expr>, rhs: Box<Expr> }
       | Let { id, name: String, value: Box<Expr>, body: Box<Expr> }
       | If  { id, cond: Box<Expr>, then_branch: Box<Expr>, else_branch: Box<Expr> }
```

## Semantics

- **`Lit`** — evaluates to the integer literal.
- **`Var`** — looks up the name first in the surrounding `Let` stack
  and then in the host's `resolved` map. When absent from both, the
  stepper yields `Suspended { reason: AwaitEffect, .. }` and the host
  is asked to supply the value via `dsl_kit_resolve`.
- **`Add` / `Mul`** — integer arithmetic on the evaluated children.
- **`Let`** — evaluates `value`, pushes `(name, value)` on the binding
  stack, evaluates `body`, pops the binding.
- **`If`** — evaluates `cond`; non-zero picks `then_branch`, else
  `else_branch`.

## Why unbound variables suspend

Unlike `flow-dsl` where `Call` nodes are explicit effect boundaries,
`expr` is a pure evaluator whose only reason to yield is a lookup
failure. The host translates each unresolved `Var` into an
`AwaitEffect` suspension so an MCP client can supply the missing
binding without the DSL having to model "effects" as first-class
nodes. This exercises the same suspend / resume contract from a
different direction.

## Node ids and traversal

Every variant carries a stable `id: NodeId`. `#[derive(DslNode)]`
generates `Walk` / `WalkMut` alongside the enum, so `expr.walk(...)` /
`expr.find_by_id(...)` work uniformly across variants.

The `pretty(expr)` free function uses `Walk` to produce the indented
tree the `dsl_kit_ast` MCP tool returns.

## Writing an expr program

```rust,ignore
use dsl_kit::IdGen;
use expr_dsl::Expr;

let ids = IdGen::new();
let program = Expr::Add {
    id: ids.node(),
    lhs: Box::new(Expr::Lit { id: ids.node(), value: 10 }),
    rhs: Box::new(Expr::Var { id: ids.node(), name: "x".into() }),
};
// Evaluating this suspends on `Var "x"` until the host resolves it.
```

For a bigger worked example see
`dsl-kit://dsl/expr/samples/demo-program`.

## Resolving variables

The `dsl_kit_resolve` tool expects `result` as an integer literal
(no default is provided when omitted). `ExprHost::step_to_done` fills
in a canned default when driving the stepper end to end: `y = 5`,
`z = 2`, and `1` for every other name.
