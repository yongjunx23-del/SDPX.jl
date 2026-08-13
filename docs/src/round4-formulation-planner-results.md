```@raw html
<!-- The source of this report is kept at
     docs/round4-formulation-planner-results.md. -->
```

```@eval
using Markdown
Markdown.parse(
    read(
        joinpath(@__DIR__, "..", "round4-formulation-planner-results.md"),
        String,
    ),
)
```
