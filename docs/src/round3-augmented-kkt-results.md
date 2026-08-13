```@raw html
<!-- The source of this report is kept at docs/round3-augmented-kkt-results.md. -->
```

```@eval
using Markdown
Markdown.parse(
    read(
        joinpath(@__DIR__, "..", "round3-augmented-kkt-results.md"),
        String,
    ),
)
```
