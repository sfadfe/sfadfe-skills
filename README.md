# sfadfe-skills

Personal Cursor Agent skills monorepo.

## Skills

| Skill | Purpose |
|-------|---------|
| [`analyze-metrics`](analyze-metrics/) | Compress TensorBoard / CSV training metrics so agents analyze without dumping long logs into context |

## Install

Copy or symlink a skill into Cursor:

```bash
ln -s "$(pwd)/analyze-metrics" ~/.cursor/skills/analyze-metrics
```

Or reference skills from this repo when working inside it.
